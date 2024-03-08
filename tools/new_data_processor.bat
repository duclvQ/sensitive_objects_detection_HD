@echo off
setlocal

set common_arg=%1
set prefix=%2
python unicode.py -p %common_arg%
python add_prefix.py -p %common_arg% -f %prefix%
python remove_duplicate_images.py -p %common_arg%

endlocal