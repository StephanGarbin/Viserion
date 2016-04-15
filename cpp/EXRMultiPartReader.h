#pragma once
#include <string>
#include <vector>

//OpenEXR
#include <ImfArray.h>
#include <ImathBox.h>
#include <ImfMultiPartInputFile.h>
#include <memory>

namespace viserion
{
	class EXRMultiPartReader
	{
	public:

		EXRMultiPartReader(const std::string& fileName,
			std::vector<std::string>& channelNames,
			Imath::Box2i& dataWindow,
			Imath::Box2i& displayWindow,
			int& numParts);

		~EXRMultiPartReader();

		bool readFloatPart2ExistingPtr(float* data,
			int multiPartIdx);

	private:
		std::shared_ptr<Imf::MultiPartInputFile> m_inputFile;
		int m_width;
		int m_height;
		int m_numChannels;
		int m_channelSize;
		Imath::Box2i m_dataWindow;
	};
}



