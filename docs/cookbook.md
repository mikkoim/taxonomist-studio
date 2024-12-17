# Contains useful tips for different situations

## Log-transforming a column

For example, with a column `dry weight (mg)` you can calculate a new column:
```
logweight = log(`dry weight (mg)`)
```


## Finding invalid images

Find outliers by grouping by individual and setting the x-axis aggregation function as "mean" and the y-axis aggregation function as `lambda x: (x.max()-x.min())/x.mean()`.


## Improving performance

The most time-consuming operation is drawing the graphs. If you want to make things faster, for example when using the image tab, set a small query to the Analysis tab to plot less images. Also setting the x and y axes to continuous values instead of discrete ones, makes the plotting faster.

## Segmentation

```bash
taxonomist-studio segment --data_folder "D:\data\MaaMet\maamet_50\Expo_2000_Ap_8" --csv_path "D:\koodia\maamet\MaaMet_images\maamet50\mm50_Image_data_cleaned.csv" --out_prefix "segmentations" --species_level
```

## Feature extraction

Feature extraction calculates feature vectors for each image in the dataset. The feature vectors are saved in a parquet file in float16. These can be later used in `dataset explore` to find similar images

```bash
taxonomist-studio embedder --data_folder "D:\data\MaaMet\maamet_50\Expo_2000_Ap_8" --csv_path "D:\koodia\maamet\MaaMet_images\maamet50\mm50_Image_data_cleaned.csv" --out_fname "features.parquet.gzip" --batch_size 128 --species_level
```

Running the embedding for a dataset of 100k images takes about 20 minutes on a CPU.


## Serving a BioDiscover dataset

`taxonomist-studio serve-dataset` starts a local server that serves the dataset. The server can be accessed at `http://localhost:5000` by default. The server can be used with Label Studio to produce additional annotations. 

```bash
taxonomist-studio serve-dataset --data_folder "D:\data\MaaMet\maamet_50\Expo_2000_Ap_8" --csv_path "D:\koodia\maamet\MaaMet_images\maamet50\mm50_Image_data_cleaned.csv" --species_level
```

## Outlier annotation with Label Studio

If you find outliers from the dataset, by for example with the Similarity Search tab in `dataset explore`, you can annotate the outliers with Label Studio. First, export the similarity search file from `dataset explore`. Then, start the server with `taxonomist-studio label-studio`, and upload the exported file to Label Studio.

```bash 
taxonomist-studio labelstudio --data_folder "D:\data\MaaMet\maamet_50\Expo_2000_Ap_8" --csv_path "D:\koodia\maamet\MaaMet_images\maamet50\mm50_Image_data_cleaned.csv" --species_level
```

The above command takes care of setting environmental variables and starting the Label Studio server.
