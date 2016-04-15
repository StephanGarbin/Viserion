#include "EXRMultiPartReader.h"

#include <iostream>
#include <algorithm>
#include <ImfPartType.h>
#include <ImfMultiPartInputFile.h>

#include <ImfChannelList.h>
#include <ImfOutputPart.h>

namespace viserion
{

	EXRMultiPartReader::EXRMultiPartReader(const std::string& fileName,
			std::vector<std::string>& channelNames,
			Imath::Box2i& dataWindow,
			Imath::Box2i& displayWindow,
			int& numParts)
	{
		try
		{
			/*Imf::Header header(displayWindow, dataWindow);

			for(auto& channel : channelNames)
			{
				header.channels().insert(channel.c_str(), Imf::Channel(Imf::FLOAT));
			}

			m_width = dataWindow.max.x - dataWindow.min.x + 1;
			m_height = dataWindow.max.y - dataWindow.min.y + 1;
			m_numChannels = channelNames.size();
			m_channelSize = width * height * channelNames.size();
			m_dataWindow = dataWindow;

			m_outputFile = std::make_shared<Imf::MultiPartOutputFile>(fileName.c_str(), &header, numParts, true);*/
		}
		catch (Iex::BaseExc& e)
		{
			std::cout << "OpenExr Error: Could not open " << fileName << std::endl;
			std::cerr << e.what() << std::endl;
		}
	}


	EXRMultiPartReader::~EXRMultiPartReader()
	{

	}

	bool EXRMultiPartReader::readFloatPart2ExistingPtr(float* data,
		int multiPartIdx)
	{
		try
		{
			
		}
		catch (Iex::BaseExc & e)
		{
			std::cerr << e.what() << std::endl;
		}
	}
}
