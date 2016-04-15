#include "ViserionImageIO.h"

#include <iostream>
#include <algorithm>
#include <sstream>
#include <boost/filesystem.hpp>

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/task_scheduler_init.h>

#include <ImathBox.h>

#include "EXRIO.h"
#include "EXRMultiPartWriter.h"
#include "ViserionImageProcessor.h"
#include "lodepng.h"

ViserionImageIO::ViserionImageIO(int val)
{
	std::cout << "CPP: New Instance..." << val << std::endl;
	m_val = val;
}


ViserionImageIO::~ViserionImageIO()
{
	std::cout << "CPP: Destroying Instance..." << m_val << std::endl;

}


bool ViserionImageIO::getNextImage()
{


}


bool ViserionImageIO::reset()
{

}

bool ViserionImageIO::readEXR(const std::string& fileName, float** destination, long& numRows, long& numCols, long& numChannels)
{
	Imath::Box2i dataWindow;
	Imath::Box2i displayWindow;

	int numRows_i;
	int numCols_i;
	int numChannels_i;
	std::vector<std::string> sortedChannelNames;

	viserion::EXRIO ioInstance(fileName);

	bool status = ioInstance.readFloatToNewPtr(destination,
			numRows_i, numCols_i, numChannels_i,
			sortedChannelNames,
			dataWindow,
			displayWindow);

	if(!status)
	{
		return false;
	}

	numRows = numRows_i;
	numCols = numCols_i;
	numChannels = numChannels_i;

	return true;
}

void loadPNG(const std::string& fileName, std::vector<float>& pixels, int& width, int& height)
{
		std::vector<unsigned char> png;
		std::vector<unsigned char> rawImage; //the raw pixels

		unsigned int lWidth;
		unsigned int lHeight;
		//load and decode
		lodepng::load_file(png, fileName);
		unsigned error = lodepng::decode(rawImage, lWidth, lHeight, png);

		width = lWidth;
		height = lHeight;

		pixels.resize(width * height * 3);

		//Handle errors
		if (error)
		{
			std::cout << "Decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
			std::cout << "Could not read [ " << fileName << " ].";
			return;
		}

		for (size_t i = 0; i < rawImage.size(); i += 4)
		{
			for (size_t c = 0; c < 3; ++c)
			{
				pixels[c * width * height + i / 4] = (float)rawImage[i + c];
			}
		}
}

bool createDataSetCache(const std::string& datasetName, const std::string& directory,
	const std::string& cacheDirectory, bool recursive, bool normalise, bool whiten)
{
	boost::filesystem::path filePath(directory);
	boost::filesystem::path cachePath(cacheDirectory);

	try
	{
		//1. Make sure all directories are okay before proceeding
		if(!boost::filesystem::exists(filePath))
		{
			std::cout << "Error in cache creation, directory [ " << directory << " ] does not exist." << std::endl; 
			return false;
		}

		if(!boost::filesystem::exists(cachePath))
		{
			std::cout << "Error in cache creation, directory [ " << cacheDirectory << " ] does not exist." << std::endl; 
			return false;
		}

		if(boost::filesystem::is_regular_file(filePath))
		{
			std::cout << "Error in cache creation, directory [ " << directory << " ] is a file rather than a directory." << std::endl; 
			return false;
		}

		if(boost::filesystem::is_regular_file(cachePath))
		{
			std::cout << "Error in cache creation, directory [ " << cacheDirectory << " ] is a file rather than a directory." << std::endl; 
			return false;
		}

		//2. Discover all files in a directory
		std::vector<boost::filesystem::path> imagePaths;

		if(recursive)
		{
			for(boost::filesystem::recursive_directory_iterator it(filePath); it != boost::filesystem::recursive_directory_iterator(); ++it)
			{
				if(it->path().extension() == ".png" || it->path().extension() == ".exr")
				{
					imagePaths.push_back(it->path());
				}
			}
		}
		else
		{
			for(boost::filesystem::directory_iterator it(filePath); it != boost::filesystem::directory_iterator(); ++it)
			{
				if(it->path().extension() == ".png" || it->path().extension() == ".exr")
				{
					imagePaths.push_back(it->path());
				}
			}
		}

		std::sort(imagePaths.begin(), imagePaths.end());

		std::cout << "Found " << imagePaths.size() << " images for cache." << std::endl;

		//3. Create Multi-Part cache archive

		//determine sizes & channels from first discovered image
		std::vector<std::string> channelNames;
		Imath::Box2i dataWindow;
		Imath::Box2i displayWindow;

		if(imagePaths[0].extension() == ".png")
		{
			std::cout << "Reading PNG.." << std::endl;
			channelNames.push_back("R");
			channelNames.push_back("G");
			channelNames.push_back("B");
			int lWidth;
			int lHeight;

			std::vector<float> temp;

			loadPNG(imagePaths[0].string(), temp, lWidth, lHeight);

			dataWindow.min.x = 0;
			dataWindow.min.y = 0;

			dataWindow.max.x = lWidth - 1;
			dataWindow.max.y = lHeight - 1;

			displayWindow = dataWindow; // pngs cannot express anything else
		}
		else
		{
			std::cout << "Reading Exr " << std::endl;
			int numRows_i;
			int numCols_i;
			int numChannels_i;

			viserion::EXRIO ioInstance(imagePaths[0].string());

			float* temp;

			bool status = ioInstance.readFloatToNewPtr(&temp,
					numRows_i, numCols_i, numChannels_i,
					channelNames,
					dataWindow,
					displayWindow);

			delete[] temp;
		}

		int width = dataWindow.max.x - dataWindow.min.x + 1;
		int height = dataWindow.max.y - dataWindow.min.y + 1;
		int numChannels = channelNames.size();

		std::cout << width << ", " << height << ",  " << numChannels << std::endl;

		std::stringstream ss;
		ss << cacheDirectory << "/" << datasetName << ".exr";

		viserion::EXRMultiPartWriter archive(ss.str(), channelNames, 
			dataWindow, displayWindow,
			imagePaths.size());

		ss.clear();

		tbb::task_scheduler_init init(1);

		//4. Iterate over all files in path and add to cache
		tbb::parallel_for(tbb::blocked_range<size_t>(0, imagePaths.size()),
			[&](const tbb::blocked_range<size_t>& r)
		{
			std::vector<float> mem(width * height * numChannels);
			std::string currentPath;
			for(size_t i = r.begin(); i != r.end(); ++i)
			{
				currentPath = imagePaths[i].string();

				Imath::Box2i tempDataWindow;
				Imath::Box2i tempDisplayWindow;
				std::vector<std::string> tempChannelNames;
				int tempNumRows;
				int tempNumCols;
				int tempNumChannels;

				if(imagePaths[i].extension() == ".png")
				{
					loadPNG(currentPath, mem, tempNumRows, tempNumCols);
				}
				else
				{
					viserion::EXRIO ioInstance(currentPath);

					bool status = ioInstance.readFloatToExistingPtr(&mem[0],
						tempNumRows, tempNumCols, tempNumChannels,
						tempChannelNames,
						tempDataWindow,
						tempDisplayWindow);
				}

				if(normalise)
				{
					viserion::normaliseVector<>(&mem[0], mem.size());
				}

				archive.writeFloatPart(&mem[0], i);	
			}
		});


		//5. Close cache & return

		return true;
	}
	catch(const boost::filesystem::filesystem_error& e)
	{
		std::cout << "Error in IO during dataset creation: " << e.what() << std::endl;
	}

	return true;
}