TensorFlow-Text-Generator:
"cb.txt" dosyasi, cumhurbaskanligi sitesinden alinan, cumhurbaskaninin yaklasik son 50 konusmasinin birlestirilmis text halidir.
https://www.tccb.gov.tr/

"hp_text.txt" dosyası ise "Harry Potter and the Sorcerer's Stone" isimli kitabın text halidir.
http://www.glozman.com/TextPages/Harry%20Potter%201%20-%20Sorcerer's%20Stone.txt

Ipython dosyası(.ipynb), train edilmis halidir

python 3.6 uzerinde yazilmistir.

Dependencies:
tensorflow 1.5.0
numpy 1.14.1

Guclu bir GPU üzerinde calistirmanizi tavsiye ederim.

"output_cb.txt" dosyasi ise 50 epoch boyunca modelin urettigi outputlardir. Yaklasik olarak 50 x 3000 = 150000 characterden olusan bir txt dosyasidir. 


fork ettigim bu link'e de bakabilirsiniz. cok guzel hazirlanmis bir .py dosyasi.
Source: https://gist.github.com/MBoustani/437cea275fa9d40c9e60eac9ba71456c

git clone https://github.com/MuhammedBuyukkinaci/TensorFlow-Text-Generator.git
cd ./TensorFlow-Text-Generator
python tensorflow_text_generator.py
