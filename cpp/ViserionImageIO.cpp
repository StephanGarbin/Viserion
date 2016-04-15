#include "ViserionImageIO.h"

#include <iostream>
#include <algorithm>
#include <sstream>
#include <boost/filesystem.hpp>

#include "EXRIO.h"
#include "EXRMultiPartWriter.h"
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

	viserion::EXRIO ioInstance(fileName);

	bool status = ioInstance.readFloatToNewPtr(destination,
			numRows_i, numCols_i, numChannels_i,
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

void loadPNG(const std::string& fileName, std::vector<float>& pixels)
{
	/*std::vector<unsigned char> png;
		std::vector<unsigned char> rawImage; //the raw pixels
		unsigned width, height;

		//load and decode
		lodepng::load_file(png, file);
		unsigned error = lodepng::decode(rawImage, width, height, png);

		//Handle errors
		if (error)
		{
			std::cout << "Decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
			std::cout << "Could not read [ " << file << " ].";
			return false;
		}
		

		Denoise::Dimension dim(width, height);
		
		initialise(dim, Denoise::Image::FLOAT_4);

		for (index_t i = 0; i < rawImage.size(); i += 4)
		{
			for (index_t c = 0; c < 4; ++c)
			{
				setPixel(c, i / 4, (float)rawImage[i + c]);
			}
		}

		return true;*/
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
		std::stringstream ss;
		ss << cacheDirectory << "/" << datasetName << ".exr";
		//viserion::EXRMultiPartWriter archive(ss.str())
		ss.clear();

		/*
		const std::string& fileName,
			const std::vector<std::string>& channelNames,
			Imath::Box2i& dataWindow,
			Imath::Box2i& displayWindow,
			int numParts*/

		//4. Iterate over all files in path and add to cache



		//5. Close cache & return

		return true;
	}
	catch(const boost::filesystem::filesystem_error& e)
	{
		std::cout << "Error in IO during dataset creation: " << e.what() << std::endl;
	}

	return true;
}