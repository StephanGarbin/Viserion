#pragma once
#include <string>
#include <vector>

//OpenEXR
#include <ImfArray.h>
#include <ImathBox.h>
#include <ImfMultiPartOutputFile.h>
#include <memory>

namespace viserion
{
	class EXRMultiPartWriter
	{
	public:

		EXRMultiPartWriter(const std::string& fileName,
			const std::vector<std::string>& channelNames,
			Imath::Box2i& dataWindow,
			Imath::Box2i& displayWindow,
			Imf::Compression compression,
			int numParts);

		~EXRMultiPartWriter();

		bool writeFloatPart(float* data,
			int multiPartIdx);

	private:
		std::shared_ptr<Imf::MultiPartOutputFile> m_outputFile;
		int m_width;
		int m_height;
		int m_numChannels;
		int m_channelSize;
		std::vector<std::string> m_channelNames;
		Imath::Box2i m_dataWindow;
	};
}



