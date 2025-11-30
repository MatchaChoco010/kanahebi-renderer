# Kanahebi CLI

kanahebi-cliはHydraレンダラープラグインのhdKanahebiを使用したコマンドラインのレンダラーのインターフェースです。
コマンドラインからOpenUSDファイル形式を指定することで、OptiX 9を使用してGPUレンダリングを行います。

```
Usage: kanahebi-cli [options] <scene.usd>
Options:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Output image directory path (default: output/)
  -w WIDTH, --width WIDTH
                        Output image width (default: render settings in OpenUSD file)
  -t HEIGHT, --height HEIGHT
                        Output image height (default: render settings in OpenUSD file)
  -s SAMPLES, --samples SAMPLES
                        Number of samples per pixel (default: 64)
  --start-frame START_FRAME
                        Start frame for animation rendering (default: render settings in OpenUSD file)
  --end-frame END_FRAME
                        End frame for animation rendering (default: render settings in OpenUSD file)
```
