#include "EXRMultiPartWriter.h"

#include <vector>
#include <iostream>
#include <algorithm>
#include <ImfPartType.h>
#include <ImfMultiPartOutputFile.h>

#include <ImfChannelList.h>
#include <ImfOutputPart.h>

namespace viserion
{

	EXRMultiPartWriter::EXRMultiPartWriter(const std::string& fileName,
		const std::vector<std::string>& channelNames,
		Imath::Box2i& dataWindow,
		Imath::Box2i& displayWindow,
		Imf::Compression compression,
		int numParts)
	{
		try
		{
			std::vector<Imf::Header> headers;
			for(size_t i = 0; i < numParts; ++i)
			{
				headers.emplace_back(displayWindow, dataWindow);

				for(auto& channel : channelNames)
				{
					headers[i].channels().insert(channel.c_str(), Imf::Channel(Imf::FLOAT));
				}
				headers[i].setName(std::to_string(i));
				headers[i].setType(Imf::SCANLINEIMAGE);
				headers[i].compression() = compression;
			}


			m_width = dataWindow.max.x - dataWindow.min.x + 1;
			m_height = dataWindow.max.y - dataWindow.min.y + 1;
			m_numChannels = channelNames.size();
			m_channelSize = m_width * m_height;
			m_dataWindow = dataWindow;
			m_channelNames = channelNames;

			m_outputFile = std::make_shared<Imf::MultiPartOutputFile>(fileName.c_str(), &headers[0], numParts, true);
			headers.clear();
		}
		catch (const std::exception &e)
		{
			std::cout << "OpenExr Error: Could not open " << fileName << std::endl;
			std::cerr << e.what() << std::endl;
		}
	}


	EXRMultiPartWriter::~EXRMultiPartWriter()
	{

	}

	bool EXRMultiPartWriter::writeFloatPart(float* data,
		int multiPartIdx)
	{
		try
		{
			Imf::FrameBuffer frameBuffer;

			int currentChannel = 0;
			for(auto& channel : m_channelNames)
			{
				frameBuffer.insert(channel.c_str(),
					Imf::Slice(Imf::FLOAT,
					(char *)(&data[currentChannel * m_channelSize] - m_dataWindow.min.x - m_dataWindow.min.y * m_width),
					sizeof (data[currentChannel * m_channelSize]),
					sizeof (data[currentChannel * m_channelSize]) * m_width));
				++currentChannel;
			}

			Imf::OutputPart	part(*m_outputFile, multiPartIdx);
			part.setFrameBuffer(frameBuffer);
			part.writePixels(m_height);

		}
		catch (const std::exception &e)
		{
			std::cerr << e.what() << std::endl;
		}
	}
}
