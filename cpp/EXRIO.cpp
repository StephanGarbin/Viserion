#include "EXRIO.h"

#include <iostream>
#include <algorithm>
#include <ImfPartType.h>
#include <ImfInputFile.h>
#include <ImfMultiPartInputFile.h>

#include <ImfMultiPartOutputFile.h>

#include <ImfChannelList.h>
#include <ImfInputPart.h>
#include <ImfOutputPart.h>

namespace viserion
{

	EXRIO::EXRIO(const std::string& fileName)
	{
		try
		{
			if(!fileName.empty())
			{
				m_inputFile = std::make_shared<Imf::InputFile>(fileName.c_str());
			}
		}
		catch (Iex::BaseExc& e)
		{
			std::cout << "OpenExr Error: Could not open " << fileName << std::endl;
			std::cerr << e.what() << std::endl;
		}
	}


	EXRIO::~EXRIO()
	{

	}

	bool EXRIO::writeFloat(const std::string& fileName,
		float* data,
		const std::vector<std::string>& channelNames,
		Imath::Box2i& dataWindow,
		Imath::Box2i& displayWindow,
		Imf::Compression compression,
		int numParts,
		int multiPartIdx)
	{
		try
		{
			Imf::Header header(displayWindow, dataWindow);

			for(auto& channel : channelNames)
			{
				header.channels().insert(channel.c_str(), Imf::Channel(Imf::FLOAT));
			}

			header.compression() = compression;

			Imf::FrameBuffer frameBuffer;

			int width = dataWindow.max.x - dataWindow.min.x + 1;
			int height = dataWindow.max.y - dataWindow.min.y + 1;

			int currentChannel = 0;
			int channelSize = width * height * channelNames.size();
			for(auto& channel : channelNames)
			{
				frameBuffer.insert(channel.c_str(),
					Imf::Slice(Imf::FLOAT,
					(char *)(&data[currentChannel * channelSize] - dataWindow.min.x - dataWindow.min.y * width),
					sizeof (data[currentChannel * channelSize]),
					sizeof (data[currentChannel * channelSize]) * width));
				++currentChannel;
			}


			if(multiPartIdx >= 0)
			{
				// we can copy data over from first header
				Imf::MultiPartOutputFile outputFile(fileName.c_str(), &header, numParts, true);

				Imf::OutputPart	part(outputFile, multiPartIdx);
				part.setFrameBuffer(frameBuffer);
				part.writePixels(height);
			}
			else
			{
				Imf::OutputFile outputFile(fileName.c_str(), header);

				outputFile.setFrameBuffer(frameBuffer);
				outputFile.writePixels(height);
			}
		}
		catch (Iex::BaseExc & e)
		{
			std::cerr << e.what() << std::endl;
		}
	}

	int EXRIO::width()
	{
		return m_inputFile->header().dataWindow().max.x - m_inputFile->header().dataWindow().min.x + 1;
	}
	
	int EXRIO::height()
	{
		return m_inputFile->header().dataWindow().max.y - m_inputFile->header().dataWindow().min.y + 1;
	}

	int EXRIO::numChannels()
	{
		const Imf::ChannelList& channels = m_inputFile->header().channels();

		int lNumChannels = 0;
		for (Imf::ChannelList::ConstIterator i = channels.begin(); i != channels.end(); ++i)
		{
			++lNumChannels;
		}

		return lNumChannels;
	}

	bool EXRIO::readFloatToExistingPtr(float* data,
		int &width, int &height, int& numChannels,
		std::vector<std::string>& sortedChannelNames,
		Imath::Box2i& dataWindow,
		Imath::Box2i& displayWindow,
		int multiPartIdx)
	{
		sortedChannelNames.clear();
		try
		{
 			displayWindow = m_inputFile->header().displayWindow();
			dataWindow = m_inputFile->header().dataWindow();

			width = dataWindow.max.x - dataWindow.min.x + 1;
			height = dataWindow.max.y - dataWindow.min.y + 1;

			//Make sure that we can handle empty images correctly
			if (width * height < 1)
			{
				return false;
			}

			Imf::FrameBuffer frameBuffer;

			const Imf::ChannelList &channels = m_inputFile->header().channels();

			numChannels = 0;
			std::vector<std::string> channelNames;
			for (Imf::ChannelList::ConstIterator i = channels.begin(); i != channels.end(); ++i)
			{
				channelNames.push_back(std::string(i.name()));
				++numChannels;
			}

			bool isRGB = false;
			bool isRGBA = false;
			bool isYUV = false;

			if(numChannels == 4
				&& std::find(channelNames.begin(), channelNames.end(), "R") != channelNames.end()
				&& std::find(channelNames.begin(), channelNames.end(), "G") != channelNames.end()
				&& std::find(channelNames.begin(), channelNames.end(), "B") != channelNames.end()
				&& std::find(channelNames.begin(), channelNames.end(), "A") != channelNames.end())
			{
				isRGBA = true;
				sortedChannelNames.push_back("R");
				sortedChannelNames.push_back("G");
				sortedChannelNames.push_back("B");
				sortedChannelNames.push_back("A");
			}
			else if (numChannels == 3
				&& std::find(channelNames.begin(), channelNames.end(), "R") != channelNames.end()
				&& std::find(channelNames.begin(), channelNames.end(), "G") != channelNames.end()
				&& std::find(channelNames.begin(), channelNames.end(), "B") != channelNames.end())
			{
				isRGB = true;
				sortedChannelNames.push_back("R");
				sortedChannelNames.push_back("G");
				sortedChannelNames.push_back("B");
			}
			else if (numChannels == 3
				&& std::find(channelNames.begin(), channelNames.end(), "Y") != channelNames.end()
				&& std::find(channelNames.begin(), channelNames.end(), "U") != channelNames.end()
				&& std::find(channelNames.begin(), channelNames.end(), "V") != channelNames.end())
			{
				isYUV = true;
				sortedChannelNames.push_back("Y");
				sortedChannelNames.push_back("U");
				sortedChannelNames.push_back("V");
			}
			int channelSize = width * height;

			if(isRGBA || isRGB || isYUV)
			{
				for(int i = 0; i < sortedChannelNames.size(); ++i)
				{
					frameBuffer.insert(sortedChannelNames[i].c_str(), 
						Imf::Slice(Imf::FLOAT,
							(char *)(&data[channelSize * i] -
							dataWindow.min.x -
							dataWindow.min.y * width),
							sizeof (data[0]) * 1,
							sizeof (data[0]) * width,
							1, 1,
							0.0f));
				}
			}
			else //read channels in order returned by the Exr libraries (alphabetical)
			{
				int currentChannel = 0;
				for (Imf::ChannelList::ConstIterator i = channels.begin(); i != channels.end(); ++i)
				{
					const Imf::Channel &channel = i.channel();

					if(channel.type != Imf::FLOAT)
					{
						std::cout << "ERROR: Channel [ " << i.name() << " ] has type other than float: " << channel.type << std::endl;
						delete[] data;
						return false;
					}

					frameBuffer.insert(i.name(), 
						Imf::Slice(channel.type,
							(char *)(&data[channelSize * currentChannel] -
							dataWindow.min.x -
							dataWindow.min.y * width),
							sizeof (data[0]) * 1,
							sizeof (data[0]) * width,
							channel.xSampling, channel.ySampling,
							0.0f));

					++currentChannel;
					sortedChannelNames.push_back(i.name());
				}
			}

			m_inputFile->setFrameBuffer(frameBuffer);
			m_inputFile->readPixels(dataWindow.min.y, dataWindow.max.y);
		}
		catch (Iex::BaseExc & e)
		{
			std::cerr << e.what() << std::endl;
			return false;
		}

		return true;
	}


	bool EXRIO::readFloatToNewPtr(float** data,
		int &width, int &height, int& numChannels,
		std::vector<std::string>& sortedChannelNames,
		Imath::Box2i& dataWindow,
		Imath::Box2i& displayWindow,
		int multiPartIdx)
	{
		sortedChannelNames.clear();
		try
		{
 			displayWindow = m_inputFile->header().displayWindow();
			dataWindow = m_inputFile->header().dataWindow();

			width = dataWindow.max.x - dataWindow.min.x + 1;
			height = dataWindow.max.y - dataWindow.min.y + 1;

			//Make sure that we can handle empty images correctly
			if (width * height < 1)
			{
				return false;
			}

			Imf::FrameBuffer frameBuffer;

			const Imf::ChannelList &channels = m_inputFile->header().channels();

			numChannels = 0;
			std::vector<std::string> channelNames;
			for (Imf::ChannelList::ConstIterator i = channels.begin(); i != channels.end(); ++i)
			{
				channelNames.push_back(std::string(i.name()));
				++numChannels;
			}

			bool isRGB = false;
			bool isRGBA = false;
			bool isYUV = false;

			if(numChannels == 4
				&& std::find(channelNames.begin(), channelNames.end(), "R") != channelNames.end()
				&& std::find(channelNames.begin(), channelNames.end(), "G") != channelNames.end()
				&& std::find(channelNames.begin(), channelNames.end(), "B") != channelNames.end()
				&& std::find(channelNames.begin(), channelNames.end(), "A") != channelNames.end())
			{
				isRGBA = true;
				sortedChannelNames.push_back("R");
				sortedChannelNames.push_back("G");
				sortedChannelNames.push_back("B");
				sortedChannelNames.push_back("A");
			}
			else if (numChannels == 3
				&& std::find(channelNames.begin(), channelNames.end(), "R") != channelNames.end()
				&& std::find(channelNames.begin(), channelNames.end(), "G") != channelNames.end()
				&& std::find(channelNames.begin(), channelNames.end(), "B") != channelNames.end())
			{
				isRGB = true;
				sortedChannelNames.push_back("R");
				sortedChannelNames.push_back("G");
				sortedChannelNames.push_back("B");
			}
			else if (numChannels == 3
				&& std::find(channelNames.begin(), channelNames.end(), "Y") != channelNames.end()
				&& std::find(channelNames.begin(), channelNames.end(), "U") != channelNames.end()
				&& std::find(channelNames.begin(), channelNames.end(), "V") != channelNames.end())
			{
				isYUV = true;
				sortedChannelNames.push_back("Y");
				sortedChannelNames.push_back("U");
				sortedChannelNames.push_back("V");
			}

			*data = new float[width * height * numChannels];

			int channelSize = width * height;

			if(isRGBA || isRGB || isYUV)
			{
				for(int i = 0; i < sortedChannelNames.size(); ++i)
				{
					frameBuffer.insert(sortedChannelNames[i].c_str(), 
						Imf::Slice(Imf::FLOAT,
							(char *)(&(*data)[channelSize * i] -
							dataWindow.min.x -
							dataWindow.min.y * width),
							sizeof ((*data)[0]) * 1,
							sizeof ((*data)[0]) * width,
							1, 1,
							0.0f));
				}
			}
			else //read channels in order returned by the Exr libraries (alphabetical)
			{
				int currentChannel = 0;
				for (Imf::ChannelList::ConstIterator i = channels.begin(); i != channels.end(); ++i)
				{
					const Imf::Channel &channel = i.channel();

					if(channel.type != Imf::FLOAT)
					{
						std::cout << "ERROR: Channel [ " << i.name() << " ] has type other than float: " << channel.type << std::endl;
						delete[] data;
						return false;
					}

					frameBuffer.insert(i.name(), 
						Imf::Slice(channel.type,
							(char *)(&(*data)[channelSize * currentChannel] -
							dataWindow.min.x -
							dataWindow.min.y * width),
							sizeof ((*data)[0]) * 1,
							sizeof ((*data)[0]) * width,
							channel.xSampling, channel.ySampling,
							0.0f));

					++currentChannel;
					sortedChannelNames.push_back(i.name());
				}
			}

			m_inputFile->setFrameBuffer(frameBuffer);
			m_inputFile->readPixels(dataWindow.min.y, dataWindow.max.y);
		}
		catch (Iex::BaseExc & e)
		{
			std::cerr << e.what() << std::endl;
			return false;
		}

		return true;
	}


	bool readFloatEXR(const std::string& fileName,
			float** data,
			int &width, int &height, int &numChannels,
			Imath::Box2i& dataWindow,
			Imath::Box2i& displayWindow)
	{
		try
		{
			Imf::InputFile file(fileName.c_str());

 			displayWindow = file.header().displayWindow();
			dataWindow = file.header().dataWindow();

			width = dataWindow.max.x - dataWindow.min.x + 1;
			height = dataWindow.max.y - dataWindow.min.y + 1;

			//Make sure that we can handle empty images correctly
			if (width * height < 1)
			{
				return false;
			}

			Imf::FrameBuffer frameBuffer;

			const Imf::ChannelList &channels = file.header().channels();

			numChannels = 0;
			std::vector<std::string> channelNames;
			std::vector<std::string> sortedChannelNames;
			for (Imf::ChannelList::ConstIterator i = channels.begin(); i != channels.end(); ++i)
			{
				channelNames.push_back(std::string(i.name()));
				++numChannels;
			}

			bool isRGB = false;
			bool isRGBA = false;
			bool isYUV = false;

			if(numChannels == 4
				&& std::find(channelNames.begin(), channelNames.end(), "R") != channelNames.end()
				&& std::find(channelNames.begin(), channelNames.end(), "G") != channelNames.end()
				&& std::find(channelNames.begin(), channelNames.end(), "B") != channelNames.end()
				&& std::find(channelNames.begin(), channelNames.end(), "A") != channelNames.end())
			{
				isRGBA = true;
				sortedChannelNames.push_back("R");
				sortedChannelNames.push_back("G");
				sortedChannelNames.push_back("B");
				sortedChannelNames.push_back("A");
			}
			else if (numChannels == 3
				&& std::find(channelNames.begin(), channelNames.end(), "R") != channelNames.end()
				&& std::find(channelNames.begin(), channelNames.end(), "G") != channelNames.end()
				&& std::find(channelNames.begin(), channelNames.end(), "B") != channelNames.end())
			{
				isRGB = true;
				sortedChannelNames.push_back("R");
				sortedChannelNames.push_back("G");
				sortedChannelNames.push_back("B");
			}
			else if (numChannels == 3
				&& std::find(channelNames.begin(), channelNames.end(), "Y") != channelNames.end()
				&& std::find(channelNames.begin(), channelNames.end(), "U") != channelNames.end()
				&& std::find(channelNames.begin(), channelNames.end(), "V") != channelNames.end())
			{
				isYUV = true;
				sortedChannelNames.push_back("Y");
				sortedChannelNames.push_back("U");
				sortedChannelNames.push_back("V");
			}

			*data = new float[width * height * numChannels];

			int channelSize = width * height;

			if(isRGBA || isRGB || isYUV)
			{
				for(int i = 0; i < sortedChannelNames.size(); ++i)
				{
					frameBuffer.insert(sortedChannelNames[i].c_str(), 
						Imf::Slice(Imf::FLOAT,
							(char *)(&(*data)[channelSize * i] -
							dataWindow.min.x -
							dataWindow.min.y * width),
							sizeof ((*data)[0]) * 1,
							sizeof ((*data)[0]) * width,
							1, 1,
							0.0f));
				}
			}
			else //read channels in order returned by the Exr libraries (alphabetical)
			{
				std::cout << "detected no known format" << std::endl;
				int currentChannel = 0;
				for (Imf::ChannelList::ConstIterator i = channels.begin(); i != channels.end(); ++i)
				{
					const Imf::Channel &channel = i.channel();

					if(channel.type != Imf::FLOAT)
					{
						std::cout << "ERROR: Channel [ " << i.name() << " ] has type other than float: " << channel.type << std::endl;
						delete[] data;
						return false;
					}

					frameBuffer.insert(i.name(), 
						Imf::Slice(channel.type,
							(char *)(&(*data)[channelSize * currentChannel] -
							dataWindow.min.x -
							dataWindow.min.y * width),
							sizeof ((*data)[0]) * 1,
							sizeof ((*data)[0]) * width,
							channel.xSampling, channel.ySampling,
							0.0f));

					++currentChannel;
				}
			}

			file.setFrameBuffer(frameBuffer);
			file.readPixels(dataWindow.min.y, dataWindow.max.y);
		}
		catch (Iex::BaseExc & e)
		{
			std::cerr << e.what() << std::endl;
			return false;
		}

		return true;
	}
}
