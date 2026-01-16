Week 3A: Black Hole Data Curation Summary

During this phase, we finished preparing the input data and ran the Python script that generates the final blackhole\_masses.csv dataset.

1\. Initial Dataset: seed\_masses.csv

We started by manually building the Ground Truth file:

Location: data/metadata/seed\_masses.csv

We filled in the required columns:





galaxy\_name — the primary/common galaxy name





log\_mbh — the published black hole mass in log₁₀(M/M☉)





We populated the file with about 80–100 galaxies and their published log₁₀(MBH) values.

2\. Running the Curation Script

After preparing the seed file, we ran the Python script data\_curation\_script.py, which handled:





Coordinate lookup using astroquery.ned to pull RA and Dec for each galaxy.





Image retrieval by using those coordinates to create placeholder images saved in data/real\_images/.





We executed the script from the root of the repository after installing the needed packages:

pip install pandas numpy astroquery requests

python data\_curation\_script.py



3\. Results

Running the script produced:





data/metadata/blackhole\_masses.csv — our final dataset containing galaxy names, log₁₀(MBH), RA, Dec, and image paths.





data/real\_images/ — a folder of placeholder image files (galaxy\_001.jpg, galaxy\_002.jpg, etc.) ready for feature extraction.





This wrapped up Week 3A and set everything in place for starting the geometry/feature extraction work in Week 3B.

