package = "ViserionImageIO"
version = "0.scm-1"

source = {
   url = "git://github.com/clementfarabet/graphicsmagick",
}

description = {
   summary = "",
   detailed = [[

   ]],
   homepage = "",
   license = ""
}

dependencies = {
   "sys >= 1.0",
   "torch >= 7.0",
   "image >= 1.0",
}

build = {
   type = "builtin",
   modules = {
      ['graphicsmagick.init'] = 'init.lua',
      ['graphicsmagick.convert'] = 'convert.lua',
      ['graphicsmagick.info'] = 'info.lua',
      ['graphicsmagick.exif'] = 'exif.lua',
      ['graphicsmagick.Image'] = 'Image.lua',
   }
}