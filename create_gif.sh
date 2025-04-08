
rm position.gif
rm temp-map.gif
rm beta.gif
rm policy.gif
ffmpeg -i output/%03d.png position.gif
ffmpeg -i output/temp-map-%03d.png temp-map.gif
ffmpeg -i output/policy_episode_%03d.png policy.gif