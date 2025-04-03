
rm position.gif
rm temp-map.gif
ffmpeg -i output/%03d.png position.gif
ffmpeg -i output/temp-map-%03d.png temp-map.gif