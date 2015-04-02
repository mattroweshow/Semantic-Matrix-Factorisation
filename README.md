# Semantic-Matrix-Factorisation

Framework written in Python that includes: SVD, SVD++, and Semantic-Variations of SVD.

See the ./examples folder for examples of how to: (i) run an arbitrary model, (ii) tune a model using scipy's minimisation routine (with Nelder-Mead), and (iii) run a model using the tuned parameters.

All models have been set up, tuned, and tested using Movie recommendations datasets: amazon movie recommendations, movielens, and movietweetings.

These have been converted to .tsv files to enable reading. Contact me if you want the data in that format. The files need to be added to the directory ./data/datasets/amazon/.., etc.

##Semantic Mappings
The ./data/semantic_mappings directory contains mappings between item IDs (from the respective movie recommendation dataset) and DBPedia URIs.
