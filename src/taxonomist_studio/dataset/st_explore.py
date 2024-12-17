import json
import logging
from pathlib import Path
from time import sleep
import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from taxonomist_studio import tools
from PIL import ImageOps, ImageEnhance
from io import BytesIO
from pycocotools.coco import COCO
import tempfile
from scipy.spatial.distance import pdist, cosine
from sklearn.linear_model import RANSACRegressor

logging.getLogger("streamlit.runtime.caching.cache_data_api").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

NAN_NOT_ALLOWED = [
    "Max Feret Diameter",
    "Perimeter",
    "Area",
    "Holes",
    "Area+Holes",
    "Exposure Time (µs)",
    "Framerate (FPS)",
    "Light Intensity (%)",
    "Aperture",
    "Image Path",
    "Date (DD/MM/YYYY HH:MM)",
    "ROI (left)",
    "ROI (top)",
    "ROI (right)",
    "ROI (bottom)",
]

CAMERA_PARAMETERS = [
    "Exposure Time (µs)",
    "Framerate (FPS)",
    "Light Intensity (%)",
    "Aperture",
]


@st.cache_data
def dropna_uploaded(df):
    """Function for dropping NaN rows from a dataframe. Caches output"""
    l1 = len(df)
    df = df.dropna(axis=0, how="all")
    df = df.reset_index(drop=True)
    if l1 - len(df) > 0:
        st.warning(f"{l1 - len(df)} empty rows detected.")
    return df


@st.cache_data
def read_uploaded_file(uploaded_file, sep):
    """Function for reading a file. Caches output"""
    if uploaded_file.name.endswith(".csv") or uploaded_file.name.endswith(".csv.zip"):
        df = pd.read_csv(uploaded_file, encoding="ISO-8859-1", sep=sep)

    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    return df

@st.cache_data
def read_parquet(feature_file):
    feature_df = pd.read_parquet(feature_file)
    return feature_df


def read_dataframe(title, key, sep=";"):
    """Creates streamlit elements for uploading dataframe files"""
    uploaded_file = st.file_uploader(title, key=key, type=[".csv", ".xlsx", ".csv.zip"])
    sep = st.text_input("Separator", key=key + "_sep", value=sep)
    st.divider()
    if uploaded_file is not None:
        df = read_uploaded_file(uploaded_file, sep)
        df = dropna_uploaded(df)
        return df
    else:
        return None


@st.cache_resource
def calculate_histogram(df, col, bins, group):
    fig = px.histogram(
        df, x=col, nbins=bins, color=group if group != "None" else None, marginal="box"
    )
    return fig



def parse_agg(agg):
    """Parses a string into a lambda function if it starts with 'lambda'.
    Returns the function and its string
    """
    if agg.startswith("lambda"):
        return eval(agg), agg
    else:
        return agg, agg


def load_images(df,
                labels=None,
                n_images=64,
                img_size=256,
                n_cols=8,
                autocontrast=False,
                segmentation_masks=False,
                random_sample=True):
    """Uses the taxonomist_studio.tools module to display images"""
    if len(df) > n_images:
        if random_sample:
            df = df.sample(n_images)
        else:
            df = df.head(n_images)
    if labels:
        label_list = tools.get_label_list(df, labels)
    else:
        label_list = None

    fpaths = tools.load_fpaths(df,
                               st.session_state.data_folder,
                               st.session_state.dir_species_level)

    return tools.show_files(
        fpaths,
        nrow=n_images // n_cols,
        label_list=label_list,
        img_size=img_size,
        return_img=True,
        autocontrast=autocontrast,
        segmentation_coco=st.session_state.coco if segmentation_masks else None,
    )


@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode("utf-8")

@st.cache_data
def calculate_unique(df, col):
    return len(df[col].unique())
    
@st.cache_resource
def calculate_grouping(df, group_col, x, y, color, hover):
    x["agg"], x["str"] = parse_agg(x["agg"])
    y["agg"], y["str"] = parse_agg(y["agg"])

    # Aggregation is done by dict, to be able to have different
    # calculations on same column
    x_col0 = f"{x['col']} ({x['str']}) x"
    y_col0 = f"{y['col']} ({y['str']}) y"
    agg_dict = {x_col0: (x["col"], x["agg"]), y_col0: (y["col"], y["agg"])}
    x["col"] = x_col0
    y["col"] = y_col0

    if color["col"] != "None":
        color["agg"], color["str"] = parse_agg(color["agg"])
        color_col0 = f"{color['col']} ({color['str']}) color"
        agg_dict[color_col0] = (color["col"], color["agg"])
        color["col"] = color_col0

    hover["str"] = []
    for c in hover["col"]:
        agg_dict[f"{c} hover"] = (c, "unique")
        hover["str"].append(f"{c} hover")

    try:
        df = df.groupby(group_col).agg(**agg_dict)
        return df, x, y, color, hover

    except Exception as e:
        st.error("Check the aggregation functions")
        logging.exception(e)
        return None, {}, {}, {}, {}

def dataframe_graphs(df):
    if st.checkbox("Show graphs"):
        df_orig = df.copy()
        st.divider()
        st.write("### Group")
        group_select = st.checkbox("Group values?")

        # Choose aggregation column
        if group_select:
            group_col = st.multiselect(
                "Group by column",
                key="groupby_col",
                options=df.columns,
                default=df.columns[0],
            )
            if len(group_col) == 1:
                group_col = group_col[0]
                has_single_group = True

        st.write("---")
        st.write("### Graph")
        # Choose graph
        graph_type = st.selectbox("Graph type", ["Scatter plot", 
                                                "Box plot",
                                                "Histogram"])
        
        is_hist = graph_type == "Histogram"
        is_scatter = graph_type == "Scatter plot"

        if graph_type in ["Scatter plot", "Histogram"]:
            with st.expander("Marginals"):
                marginal_x = st.selectbox("Marginal X", [None,
                                                        "box",
                                                        "violin",
                                                        "rug",
                                                        "histogram"])
                if not is_hist:
                    marginal_y = st.selectbox("Marginal Y", [None,
                                                            "box",
                                                            "violin",
                                                            "rug",
                                                            "histogram"])
                if is_hist:
                    other_kwargs = {"marginal": marginal_x}
                else:
                    other_kwargs = {"marginal_x": marginal_x,
                                    "marginal_y": marginal_y}
        else:
            other_kwargs = {}
        
        with st.expander("Log axes"):
            log_x = st.checkbox("Logarithmic X axis")
            log_y = st.checkbox("Logarithmic Y axis")
            if log_x:
                other_kwargs["log_x"] = True
            if log_y:
                other_kwargs["log_y"] = True

        # Choose columns
        ccol_col, ccol_agg = st.columns(2)
        x = {}
        y = {}
        color = {}
        hover = {}
        with ccol_col:
            x["col"] = st.selectbox(
                "Select x axis column", 
                key="x_col_sel", 
                options=df.columns
            )

            if is_hist:
                y["col"] = df.columns[0]
            else:
                y["col"] = st.selectbox(
                    "Select y axis column",
                    key="y_col_sel",
                    options=df.columns
                )

            color["col"] = st.selectbox(
                "Select color column",
                key="color_col_sel",
                options=["None"] + list(df.columns),
            )

        # Aggregation function specification
        with ccol_agg:
            if group_select:
                x["agg"] = st.text_input(
                    "Pandas x-axis aggregation function", key="x_col_agg", value="mean"
                )
                if is_hist:
                    y["agg"] = "first"
                else:
                    y["agg"] = st.text_input(
                        "Pandas y-axis aggregation function", key="y_col_agg", value="mean"
                    )
                if color["col"] != "None":
                    color["agg"] = st.text_input(
                        "Pandas color aggregation function",
                        key="color_col_agg",
                        value="mean",
                    )

        hover["col"] = st.multiselect(
            "Select hover columns", key="graph_hover", options=df.columns
        )

        agg_ok = True  # Assume aggregation functions work
        if group_select:
            df, x, y, color, hover = calculate_grouping(
                df, group_col, x=x, y=y, color=color, hover=hover
            )
            if df is None:
                agg_ok = False

        else:
            x["str"] = x["col"]
            y["str"] = y["col"]
            color["str"] = color["col"]
            hover["str"] = hover["col"]

        if graph_type == "Box plot":
            px_graph = px.box
        elif graph_type == "Scatter plot":
            px_graph = px.scatter
        elif graph_type == "Histogram":
            px_graph = px.histogram
            y["col"] = None

        if agg_ok:
            if group_select:
                st.write(f"Grouped by *{group_col}* to {len(df)} groups")

            if is_hist:
                st.write("Histogram options:")
                max_bins = calculate_unique(df, x["col"])
                other_kwargs["nbins"] = st.slider("Number of bins. Set 0 for auto",
                                                    min_value=0,
                                                    max_value=max_bins,
                                                    value=0)
                if other_kwargs["nbins"] == 0:
                    other_kwargs["nbins"] = None
                
                swap_hist_axes = st.checkbox("Swap histogram axes")
                if swap_hist_axes:
                    temp = x
                    x = y
                    y = temp
            fig = px_graph(
                df,
                x=x["col"],
                y=y["col"],
                color=color["col"] if color["col"] != "None" else None,
                hover_data=hover["str"],
                labels={
                    x["col"]: x["str"],
                    y["col"]: y["str"],
                    color["col"]: color["str"] if color["col"] != "None" else None,
                },
                **other_kwargs
            )
            if is_hist:
                order_hist = st.checkbox("Order histogram")
                if order_hist:
                    if swap_hist_axes:
                        fig = fig.update_yaxes(categoryorder="total ascending")
                    else:
                        fig = fig.update_xaxes(categoryorder="total descending")

            if is_scatter:
                if st.checkbox("Open marker"):
                    fig = fig.update_traces(marker=dict(symbol="circle-open"))

            st.plotly_chart(fig, key="graph_chart", use_container_width=True)

            with st.expander("Graph dataframe"):
                dataframe_head(df, key="graph dataframe")

            st.write("## Search")
            filter_query = st.text_input("Query graph dataframe")
            if filter_query != "":
                df_search_df = df.query(filter_query)
                st.write(f"Selected {len(df_search_df)} values from {len(df)}")

                group_col_list = [group_col] if has_single_group else list(group_col)
                if group_select:
                    df_search = df_orig[
                        group_col_list + ["Image File Name", "Image Path", "run"]
                    ].merge(df_search_df, on=group_col)
                else:
                    df_search = df_orig.loc[df_search_df.index]

            else:
                df_search = df_orig

            search_col = st.selectbox("Select column", options=list(df_search.columns))

            df_search = column_filter(df_search, search_col, key="after_analysis")
            st.write(f"{len(df_search)} rows with selection")
            with st.expander("Dataframe"):
                dataframe_head(df_search, key="analysis_df")
                csv = convert_df(df_search)
                st.download_button(
                    label="Download dataframe",
                    data=csv,
                    file_name="output.csv",
                    mime="text/csv",
                )

            st.write("## Images")
            show_images(df_search, key="analysis_show_images")

            # with search_tab:
            #     search_query = st.text_input("Image search query")
            #     if search_query != "":
            #         df_search_df = df.query(search_query)
            #         st.write(f"Selected {len(df_search_df)} values from {len(df)}")
            #         with st.expander("Filtered dataframe"):
            #             st.dataframe(df_search_df)

            #         img_search_ok = True
            #     else:
            #         st.dataframe(df)

@st.cache_data
def calculate_column_filter(df, col, vals, range):
    # Filtering
    if vals:
        df = df[df[col].isin(vals)]
    elif range:
        df = df[df[col] >= range[0]]
        df = df[df[col] <= range[1]]
    return df

def column_filter(df, col, key=None):
    """Selects a column and filters a dataframe based on chosen values"""
    if df[col].dtype.kind in "iufc":
        try:
            minval = int(df[col].min())
            maxval = int(df[col].max())
            range = st.slider(
                f"Select value range from {col}", minval, maxval, (minval, maxval)
            )
            vals = None
        except ValueError:
            st.error("Can't choose values from column")
            range = None
            vals = None
    else:
        range = None
        vals = st.multiselect(f"Select values from *{col}*", df[col].unique(), key=key)

    df = calculate_column_filter(df, col, vals, range)

    return df

def image_tab_filter(df):
    """Filters the dataframe for the image showing tab"""
    col = st.selectbox("Select column for filter", df.columns)

    counts = df[col].value_counts()

    if st.checkbox("Show distributions"):
        bins = st.slider(
            "Number of bins", key="imagefilter_slider", max_value=len(counts)
        )

        fig = calculate_histogram(df, col, bins, group="None")
        st.plotly_chart(fig, key="imagefilter_chart", use_container_width=True)

    df = column_filter(df, col, key="image tab filter")

    st.write(f"{len(df)} rows with this selection")
    return df

def calculate_img_processing(img_bw, 
                             img_equalize,
                             img_contrast,
                             img_brightness):
    img = st.session_state.img
    if img_bw:
        img = img.convert("L")
    if img_equalize:
        img = ImageOps.equalize(img, mask=None)
    
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(img_contrast)

    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(img_brightness)
    
    # Download
    buf = BytesIO()
    img.save(buf, format="PNG") 
    st.session_state.img_buf = buf
    st.session_state.img_show = img

def show_images(df, key=None, random_sample=True):

    # Display options and image choosing are in different order, so a container
    image_option_container = st.container()

    # Loaded image display options
    with st.expander("Display options"):
        img_width = st.slider("Image width", 512, 2080, 1280, key=f"{key} width")
        img_contrast = st.slider("Image contrast", 0.0,2.0,1.0, key=f"{key} contrast")
        img_brightness = st.slider("Image brightness", 0.0,2.0,1.0, key=f"{key} brightness")
        img_bw = st.checkbox("Grayscale", key=f"{key} bw")
        img_equalize = st.checkbox("Equalize histogram", key=f"{key} eq")
        
        st.button("Update", on_click=calculate_img_processing, args=[img_bw,
                                                                     img_equalize,
                                                                     img_contrast,
                                                                     img_brightness],
                                                                     key=f"{key} button")

    # Image loading 
    with image_option_container.form(f"{key} Image options"):
        labels = st.multiselect("Select labels", df.columns, key=key)

        img_size = st.number_input("Image size", value=256, key=f"{key} size")
        n_images = st.number_input("Show N Images", value=64, key=f"{key} n images")

        if st.session_state.coco is not None:
            show_segment_checkbox = st.checkbox("Show segmentation masks", key=f"{key} segment")
        else:
            show_segment_checkbox = False

        with st.expander("Additional options"):
            n_cols = st.number_input("N cols", value=8, key=f"{key} n cols")
            img_autocontrast = st.checkbox("Autocontrast each image separately",
                                           key=f"{key} autocontrast")

        submitted = st.form_submit_button("Load a batch of images")
        if submitted:
            with st.spinner("Reading files..."):
                st.session_state.img = load_images(
                    df, 
                    labels=labels,
                    n_images=n_images,
                    img_size=img_size,
                    n_cols = n_cols,
                    autocontrast=img_autocontrast,
                    segmentation_masks=show_segment_checkbox,
                    random_sample=random_sample
                )
                st.session_state.img_show = st.session_state.img.copy()
                calculate_img_processing(img_bw,
                                         img_equalize,
                                         img_contrast,
                                         img_brightness)

    # Image display and download
    if "img" in st.session_state.keys():
        img = st.session_state.img_show

        if "img_buf" in st.session_state.keys():
            st.download_button("Download image",
                            data=st.session_state.img_buf.getvalue(),
                            file_name="images.png",
                            mime="image/png",
                            key=f"{key} download button")
        st.image(img, width=img_width)
    else:
        st.write("Load images first")


@st.cache_data
def load_query_image(df, query_fname):
    query_row = df[df["Image File Name"] == query_fname]
    query_fpath = tools.load_fpaths(query_row,
                                    st.session_state.data_folder,
                                    st.session_state.dir_species_level).values[0]
    return tools.show_files([query_fpath], nrow=1, return_img=True)

@st.cache_data
def load_similar_images(df, query_fname, feature_df):
    feature_query = feature_df.loc[query_fname].values
    # Group the features into an array
    X = feature_df.values
    # Calculate cosine similarity
    d_cos = np.apply_along_axis(lambda x: 1-cosine(feature_query, x), axis=1, arr=X)
    similarity_map = dict(zip(feature_df.index, d_cos))

    df["similarity"] = (df["Image File Name"]
                    .apply(lambda x: similarity_map[x] if x in similarity_map else None)
    )
    df = df.sort_values("similarity", ascending=False)
    return df

def process_similarity_csv(df, host, col="similarity"):
    dfsim = df[["Sample Name/Number", 
        "Image File Name",
        "Species Name",
        col]]
    dfsim = dfsim.assign(url=dfsim["Image File Name"]
            .apply(lambda x: f"{host}/{x}"))
    return dfsim

def outlier_detection_init(df, feature_df, col):
    """Find a filename->group map and apply it to a dataframe index
    
    Args:
        df (pd.DataFrame): The BioDiscover metadata dataframe
        feature_df (pd.DataFrame): The feature vector dataframe. Index is file name
        col (str): The column to map
    Returns:
        fname2group (dict): Mapping from file name to group
        group_index (pd.Index): feature_df index mapped to group
    """
    # Mapping from image file name to group
    fname2group = dict(zip(df["Image File Name"].values, df[col].values))

    # Remap the image-level features
    group_index = feature_df.index.map(lambda x: fname2group[x])
    return fname2group, group_index 

def calculate_mean_pairwise_distance(X):
    """Calculates the mean pairwise distances for matrix X
    If the matrix is too large, a random sample of 1000 rows is taken
    Calculates pairwise cosine distance between all rows in X and returns their mean
    
    Args:
        X (np.array): The feature matrix
    Returns:
        gd (float): The mean pairwise distance
    """
    if len(X.shape) == 2:
        if X.shape[0] > 1000:
            X = X[np.random.choice(np.arange(X.shape[0]), 1000, replace=False)]
        Xd = pdist(X, 'cosine')
        gd = Xd.mean() if len(Xd) > 1 else 1
        return gd
    return 1

@st.cache_data
def calculate_outlier_detection(df, group_col, normalization_col):
    """Performs outlier detection based on cosine similarity to group mean vectors
    
    Args:
        df (pd.DataFrame): The BioDiscover metadata dataframe
        group_col (str): The column to group by
        normalization_col (str): The column to normalize by
    Returns:
        df (pd.DataFrame): A dataframe ordered by cosine distances (or normalized distances if normalization_col is set)
    """
    df = df[df["Image File Name"].isin(st.session_state.feature_df.index)]

    feature_df = st.session_state.feature_df
    feature_df = feature_df[feature_df.index.isin(df["Image File Name"])]
    group_features = tools.group_features(df,
                                          feature_df,
                                          group_col=group_col)
    fname2group, group_index = outlier_detection_init(df, feature_df, group_col)

    # Calculate group mean pairwise distances
    if normalization_col is not None:
        fname2normalizationgroup, normalization_index = outlier_detection_init(df, feature_df, normalization_col)
        glist = normalization_index.unique()
        n = len(glist)
        group_means = {}
        pbar = st.progress(0, text="Calculating group mean distances for normalization...")
        for i, g in enumerate(glist):
            X = feature_df.iloc[np.where(normalization_index == g)[0]].values
            gd = calculate_mean_pairwise_distance(X)
            group_means[g] = gd

            if i%50 == 0:
                pbar.progress(i/n, text=f"Calculating group mean distances for normalization... {i+1}/{n}")
    
    # Calculate distances to the group mean
    d = []
    n = len(feature_df)
    pbar = st.progress(0, text="Calculating distances...")
    for i, (fname, vec) in enumerate(feature_df.iterrows()):
        group = fname2group[fname]
        group_mean_vec = group_features.loc[group]
        dg = cosine(vec.values, group_mean_vec)
        if normalization_col is not None:
            normalization_group = fname2normalizationgroup[fname]
            dg = dg / group_means[normalization_group]
        d.append(dg)
        if i%50 == 0:
            pbar.progress(i/n, text=f"Calculating distances... {i+1}/{n}")

    d_series = pd.Series(d, index=feature_df.index)
    d_series.name = "cosine_distance"
    return df.join(d_series, on="Image File Name").sort_values(by="cosine_distance", ascending=False)


def dist_from_line(x,y, a,b,c):
    """
    x,y: the point
    a,b,c: line coefficients, a*x + b*y + c = 0

    returns the signed distance from the line
    """
    d = (np.abs(a*x + b*y + c)) / (np.sqrt( a**2 + b**2))
    if y > (-a*x - c)/b:
        return d
    return -d

def ransac_ranker(df):
    """
    Re-orders the df based on the relative area difference among the largest and smallest are of the specimen
    The motivation is that if the image sequence contains images from the same specimen, all areas should be close
    to the mean. When the images contain debree, legs etc. the size difference between specimen and outlier images
    is large.

    Args:
        df (pd.DataFrame): The BioDiscover metadata dataframe
    
    Returns:
    """

    # Calculate the relative differences between area
    x = np.log(df.groupby("specimen")["Area"].mean())
    y = np.log(df.groupby("specimen")["Area"].agg(lambda x: (x.max()-x.min())/x.mean()))

    # Calculate RANSAC regression line and it's coefficients
    ransac = RANSACRegressor()
    ransac.fit(x.values.reshape(-1,1), y.values)

    a = ransac.estimator_.coef_
    b = -1
    c = ransac.estimator_.intercept_

    # Calculate signed distance from the RANSAC line
    d = [dist_from_line(x,y ,a,b,c)[0] for x,y in zip(x.values, y.values)]
    return pd.Series(data=d, index=x.index)

def check_folder(data_folder):
    """Checks if folder exists"""
    if data_folder is not None:  # Check if folder exists
        if Path(data_folder).exists() and data_folder != "":
            st.success("✔️ Folder ok!")
        else:
            st.error("❌ Invalid folder")


def click():
    with st.spinner("Wait"):
        sleep(2)


@st.cache_data
def merge_dfs(df, dfi, dfm, merge_df, merge_meta):
    """Merges dataframes for analysis"""

    # Only BioDiscover dataframe
    if df is None:
        st.error("Nothing to merge. You need at least the BioDiscover spreadsheet")
    else:
        orig_n = len(df)

    # Biodiscover and image scan
    if (df is not None) and (dfi is not None):
        # Filter df based on image filenames in dfi
        df = df[df["Image File Name"].isin(dfi["image"])]

        # All three
        if dfm is not None:
            df = df.merge(dfm, left_on=merge_df, right_on=merge_meta)
            st.success(f"Merged all three dataframes. {orig_n-len(df)} rows removed.")

        else:
            st.success(
                "Filtered BioDiscover to match Image Folder Scan data. "
                f"{orig_n-len(df)} rows removed."
            )
        return df

    # Biodiscover and metadata
    elif (df is not None) and (dfm is not None):
        df = df.merge(dfm, left_on=merge_df, right_on=merge_meta)
        st.success(
            f"Merged BioDiscover data and metadata. {orig_n-len(df)} rows removed."
        )
        return df
    else:
        st.error("Nothing to merge")

@st.cache_data
def add_additional_biodiscover_columns(df):
    """Adds additional columns to the BioDiscover dataframe"""
    return tools.add_additional_biodiscover_columns(df)

@st.cache_data
def calculate_falling_speed(df):
    return tools.calculate_falling_speed(df, top_to_bottom=not st.session_state.bd_xl)


def filter_file_element(df, key=None):
    """Creates streamlit elements for reading a filter file and filtering the dataframe
    
    Args:
        df (pd.DataFrame): The dataframe to filter
        key (str): The key for the streamlit elements, must be unique
    """
    filter_file = st.file_uploader("Upload filter file", type=[".csv"], key=key)

    if filter_file is not None:
        df_filter = read_uploaded_file(filter_file, sep=",")
        df_filter = dropna_uploaded(df_filter)
        orig_len = len(df)
        df = filter_by_another_df(df, df_filter, "Image File Name", "Image File Name")
        st.write(f"Filtered {len(df_filter)}/{orig_len} rows")
    return df

@st.cache_data
def filter_by_another_df(df, df_filter, col, filter_col):
    """Filters a dataframe based on another dataframe. Removes rows where filter_col
    values are present in the original dataframe's col
    Args:
        df (pd.DataFrame): The original dataframe
        df_filter (pd.DataFrame): The dataframe to filter by
        col (str): The column in the original dataframe
        filter_col (str): The column in the filter dataframe
    """
    return df[~df[col].isin(df_filter[filter_col])]


class BDDataFrames:
    """Class for storing three different biodiscover dataframes"""

    def __init__(self):
        self._df = None
        self._dfm = None
        self._dfi = None

        self._df_merge = None

    def set_df(self, df):
        self._df = df

    def set_dfi(self, dfi):
        self._dfi = dfi

    def set_dfm(self, dfm):
        self._dfm = dfm

    def set_df_merge(self, df_merge):
        self._df_merge = df_merge

    @property
    def has_df(self):
        return self._df is not None

    @property
    def has_dfm(self):
        return self._dfm is not None

    @property
    def has_dfi(self):
        return self._dfi is not None

    @property
    def has_df_merge(self):
        return self._df_merge is not None

    @property
    def df(self):
        if self.has_df:
            return self._df.copy()
        else:
            return None

    @property
    def dfi(self):
        if self.has_dfi:
            return self._dfi.copy()
        else:
            return None

    @property
    def dfm(self):
        if self.has_dfm:
            return self._dfm.copy()
        else:
            return None

    @property
    def df_merge(self):
        if self.has_df_merge:
            return self._df_merge.copy()
        else:
            return None


def update_df_object(df, dfi, dfm):
    if "df_object" not in st.session_state.keys():
        st.session_state.df_object = BDDataFrames()
    else:
        if not st.session_state.df_object.has_df:
            st.session_state.df_object.set_df(df)
        if not st.session_state.df_object.has_dfi:
            st.session_state.df_object.set_dfi(dfi)
        if not st.session_state.df_object.has_dfm:
            st.session_state.df_object.set_dfm(dfm)


def make_sidebar():
    """Creates the sidebar"""

    # left-to-right imaging check for speed calculations
    st.session_state.bd_xl = st.checkbox("Left-to-right imaging?", value=True)
    jpeg_files = st.checkbox("jpeg files on disk?")

    # BioDiscover spreadsheet
    df = read_dataframe("BioDiscover spreadsheet (Image data)", "read_df")

    if df is not None:
        try:
            df = add_additional_biodiscover_columns(df)
            df = calculate_falling_speed(df)
        except Exception as e:
            st.error("Error in falling speed calculations. Check the error message:")
            st.exception(e)
        if jpeg_files:
            df["Image File Name"] = df["Image File Name"].apply(lambda x: x.replace(".PNG", ".jpg"))
    

    # Image folder
    data_folder = st.text_input("Data folder", value="")
    check_folder(data_folder)
    st.session_state.dir_species_level = st.checkbox("Directory is on species level")
    dfi = read_dataframe("Image scan file", "read_dfi", sep=",")

    # Metadata
    dfm = read_dataframe("Metadata spreadsheet", "read_dfm")
    # Create the dataframe object in session state
    update_df_object(df, dfi, dfm)

    st.session_state.data_folder = data_folder  # Data folder

    # Segmentation data selection
    segmentation_file = st.file_uploader("Segmentation masks in COCO format",
                    key="segmentation_masks", type=[".json"])
    if segmentation_file is not None:
        temp_dir = tempfile.mkdtemp()
        path = os.path.join(temp_dir, segmentation_file.name)
        with open(path, "wb") as f:
            f.write(segmentation_file.getvalue())
        st.session_state.coco = COCO(path)
        st.success("Segmentation masks loaded")
    else:
        st.session_state.coco = None
    


@st.cache_data
def calculate_filter(df, drop_cols, evaluation_query, filter_query):
    if drop_cols is not None:
        df = df.drop(drop_cols, axis=1)

    if evaluation_query != "":
        df = df.eval(evaluation_query, engine="python")

    if filter_query != "":
        df = df.query(filter_query)
    return df


@st.cache_data
def calculate_dropna(df):
    df = df.dropna(axis=1, how="all")
    return df

def dataframe_filtering(df):
    orig_len = len(df)
    orig_n_col = len(df.columns)
    with st.expander("Filtering"):
        if st.checkbox("Filter by another file"):
            df = filter_file_element(df, key="analysis_filtering")

        if st.checkbox("Drop columns with all NaN"):
            df = calculate_dropna(df)
        drop_cols = st.multiselect("Drop columns", options=df.columns)
        evaluation_query = st.text_input("Calculate new columns")
        filter_query = st.text_input("Filter query")

        df = calculate_filter(df, drop_cols, evaluation_query, filter_query)

        st.divider()
        st.write("Additional single column value based filtering")
        col = st.selectbox("Select column for filter", df.columns, key="analysis_filter")
        df = column_filter(df, col)

    st.write(
        f"Using {len(df)}/{orig_len} rows | "
        f"{len(df.columns)}/{orig_n_col} columns after filtering"
    )
    return df

@st.cache_data
def calculate_df_mismatch(series_a, series_b, return_set=False):
    s = set(series_a)
    si = set(series_b)
    if return_set:
        return len(s-si), len(si-s), s,si
    else:
        return len(s-si), len(si-s)

def display_df_dfi_mismatch():
    diff_a, diff_b = calculate_df_mismatch(df_object.df["Image File Name"],
                                           df_object.dfi.image)
    if diff_a > 0 or diff_b > 0:
        st.warning(
            f"""
        Mismatches in BioDiscover and scanned images:

        Folder contains **{diff_b}** images not in spreadsheet.

        Spreadsheet contains **{diff_a}** images not in folder.

        See Quality Control tab for details.
        """
        )

def dataframe_head(df, key=None):
    """Displays first 20 values of dataframe with the option to display full dataframe"""
    if st.checkbox("Show all rows", key=key):
        st.dataframe(df)
        st.write(f"{len(df)} rows")
    else:
        st.dataframe(df.head(20))
        st.write(f"Showing {min(20, len(df))}/{len(df)} rows")

if __name__ == "__main__":
    st.set_page_config(page_title="BioDiscover Studio", layout="wide", page_icon="🐛")
    
    st.write(
        """
    # 🐛 BioDiscover Studio 📸
    """
    )
    with st.sidebar:  # Sidebar for input data
        make_sidebar()

    df_object = st.session_state.df_object
    tab_df, tab_analysis, tab_images, tab_simsearch, tab_qc = st.tabs(
        ["Data", "Analysis", "Images", "Similarity search", "Quality control"]
    )
    with tab_df:
        # Biodiscover dataframe
        st.write("### BioDiscover dataframe")
        if df_object.has_df:
            df = df_object.df
            dataframe_head(df, key="all_df")

        else:
            st.warning("BioDiscover file not selected")

        # Image folder dataframe
        st.write("### Image folder dataframe")
        if df_object.has_dfi:
            dfi = df_object.dfi
            dataframe_head(dfi, key="all_dfi")
            if df_object.has_df:  # Compare folder and spreadsheet
                display_df_dfi_mismatch()
        else:
            st.warning("Image folder file not selected")

        # Metadata dataframe
        st.write("### Metadata dataframe")
        if df_object.has_dfm:
            dfm = df_object.dfm
            dataframe_head(dfm, key="all_dfm")
        else:
            st.warning("Metadata file not selected")

        st.write("## Merge dataframes")

        # Form for metadata merging
        with st.form("Merging form"):
            if df_object.has_dfm:
                merge_col_meta = st.selectbox(
                    "Select metadata key", df_object.dfm.columns
                )
                merge_col_df = st.selectbox(
                    "Select image data key", df_object.df.columns
                )
            else:
                merge_col_meta = None
                merge_col_df = None

            submitted = st.form_submit_button("Merge")
            if submitted:
                df_merge = merge_dfs(
                    df_object.df,
                    df_object.dfi,
                    df_object.dfm,
                    merge_col_df,
                    merge_col_meta,
                )
                st.session_state.df_object.set_df_merge(df_merge)

        st.write("Merged dataset")
        if df_object.has_df_merge:
            dataframe_head(df_object.df_merge, key="all_dfmerge")
        else:
            st.warning("Dataframes not merged")

    with tab_analysis:
        df_map = {
            "Merged": st.session_state.df_object.df_merge,
            "BioDiscover": st.session_state.df_object.df,
            "Metadata": st.session_state.df_object.dfm,
            "Folder scan": st.session_state.df_object.dfi,
        }
        chosen_dataframe = st.selectbox(
            "Select dataframe", ["Merged", "BioDiscover", "Metadata", "Folder scan"]
        )

        df = df_map[chosen_dataframe]

        if df is not None:
            st.write("### Filter")
            df = dataframe_filtering(df)
            dataframe_head(df, key="show_all_filt")
            
            dataframe_graphs(df)
        else:
            st.warning("Chosen dataframe does not exist! Either upload it or merge it.")

    with tab_images:
        check_folder(st.session_state.data_folder)
        if st.session_state.df_object.has_df_merge:
            df = st.session_state.df_object.df_merge
            df = image_tab_filter(df)
            show_images(df)
        else:
            st.warning(
                "Please merge the BioDiscover spreadsheet and the Image Scan "
                "spreadsheets"
            )

    with tab_simsearch:
        st.write("Similarity search makes it possible to find "
                 "similar images based on feature vectors. "
                 "Useful especially for removing incorrect images from the dataset.")
        if st.session_state.df_object.has_df_merge:
            df = st.session_state.df_object.df_merge
            feature_file = st.file_uploader("Upload feature file", type=[".parquet.gzip", ".parquet"])

            # Feature file ok
            if feature_file is not None:

                feature_df = read_parquet(feature_file)
                if st.session_state.feature_df is None:
                    st.session_state.feature_df = feature_df
                if len(feature_df) != len(df):
                    st.warning(f"Feature file does not match the dataframe. "
                            f"Feature file contains {len(feature_df)} images "
                            f"while the dataframe contains {len(df)} images")
                # Query image selection
                query_fname = st.selectbox("Select image", feature_df.index, key="simsearch_select")

                # Load query image
                query_img = load_query_image(df, query_fname)
                st.image(query_img, width=256)

                # Similar images
                if st.button("Calculate similar images"):
                    df = df[df["Image File Name"].isin(feature_df.index)]
                    st.session_state.simsearch_df = load_similar_images(df, query_fname, feature_df)
                
                if "simsearch_df" in st.session_state:
                    show_images(st.session_state.simsearch_df, key="simsearch_images", random_sample=False)

                    # Export to Label Studio
                    st.write("### Export to Label Studio")
                    st.write("Set the host to the image server.")
                    host = st.text_input("Host", value="http://localhost:5000")
                    dfsim = process_similarity_csv(st.session_state.simsearch_df, host)
                    csv = convert_df(dfsim)
                    st.download_button(
                        label="Export Label Studio csv",
                        data=csv,
                        file_name="label_studio_tasks.csv",
                        mime="text/csv",
                    )
        else:
            st.warning("Merge the dataset first")
        if "feature_df" not in st.session_state:
            st.session_state.feature_df = None

            

    with tab_qc:
        df_map = {
            "Merged": st.session_state.df_object.df_merge,
            "BioDiscover": st.session_state.df_object.df,
        }
        st.write("## Quality check")
        chosen_dataframe = st.selectbox(
            "Select dataframe", ["Merged", "BioDiscover"], key="qc_select"
        )
        if st.button("Run Quality Check"):
            df = df_map[chosen_dataframe]

            # Missing merge files
            if df_object.has_df_merge and chosen_dataframe == "Merged":
                diff_a, diff_b, s, si = calculate_df_mismatch(
                                                df_object.df["Image File Name"],
                                                dfi.image,
                                                return_set=True)
                st.warning(
                    f"""
                Mismatches in BioDiscover and scanned images:

                Folder contains **{diff_b}** images not in spreadsheet.

                Spreadsheet contains **{diff_a}** images not in folder.
                """
                )
                with st.expander("Mismatches"):
                    if diff_a > 0:
                        df_mismatch_df = df_object.df[
                            df_object.df["Image File Name"].isin(s - si)
                        ]
                        st.write("Files in spreadsheet but not in folder")
                        st.dataframe(df_mismatch_df)
                    elif diff_b > 0:
                        df_mismatch_dfi = df_object.dfi[
                            df_object.dfi["image"].isin(si - s)
                        ]
                        st.write("Files in folder but not in spreadsheet")
                        st.dataframe(df_mismatch_dfi)
            # NaN values
            nan_ok = True
            for colname in NAN_NOT_ALLOWED:
                nans = df[colname].isna()
                if nans.any():
                    st.error(f"{nans.sum()} NaN values in *{colname}*")
                    nan_ok = False
            if nan_ok:
                st.success("No unexpected NaN values!")

            # Camera parameters
            camera_params_ok = True
            for colname in CAMERA_PARAMETERS:
                nunique = df[colname].nunique()
                if nunique != 1:
                    st.warning(f"{nunique} values found in *{colname}*")
                    one_value_ok = False
            if camera_params_ok:
                st.success("Camera parameters ok!")

            if not np.all(df["Sample Name/Number"] == df["run"]):
                st.error(
                    "Sample Name/Number does not match filename. "
                    "Run filter query``` `Sample Name/Number` != run ```to check"
                )

            if df["Image File Name"].is_unique:
                st.success("Image File Names are unique!")
            else:
                st.error("Image File Names are not unique!")

        # Outlier detection
        st.divider()
        st.write("## Outlier detection")
        if st.session_state.df_object.has_df_merge:
            
            df = st.session_state.df_object.df_merge

            st.write("## Size difference based outlier detection")
            st.write("Re-orders the df based on the relative area difference among the "
                     "largest and smallest are of the specimen. The motivation "
                     "behind this is that if the image sequence contains images from "
                     "the same specimen, all areas should be close to the mean. "
                     "When the images contain debree, legs etc. the size difference "
                     "between specimen and outlier images is large.")

            if st.button("Calculate"):
                pass

            st.write("## Visual similarity based outlier detection")
            st.write("Visual similarity outlier detection compares each image separately to the mean "
                     "feature vector of a selected group. So if the grouping column is set as "
                     "'Species Name', each image is compared to the 'mean image' of their "
                     "respective species. The most different images are returned.\n\n"
                     "Optionally you can normalize the distances by the mean distance among "
                     "all images in another group. This makes it possible to find more fine-grained differences.")
            if st.session_state.feature_df is not None:
                ood_group_col = st.selectbox("grouping column", df.columns, key="qc_outliers_group")
                if st.checkbox("Normalization"):
                    ood_norm_col = st.selectbox("Normalization column", df.columns, key="qc_outliers_normalization")
                else:
                    ood_norm_col = None
                
                if st.checkbox("Filter"):
                    df = filter_file_element(df, key="ood_filter")
                
                if st.button("Calculate"):
                    st.session_state.df_dist_sorted = calculate_outlier_detection(df, ood_group_col, ood_norm_col)
                    
                if "df_dist_sorted" in st.session_state: 
                    
                    show_images(st.session_state.df_dist_sorted, key="ood_images", random_sample=False)

                    # Export to Label Studio
                    st.write("### Export to Label Studio")
                    st.write("Set the host to the image server.")
                    host = st.text_input("Host", value="http://localhost:5000", key="ood_host")
                    dfsim = process_similarity_csv(st.session_state.df_dist_sorted, host, col="cosine_distance")
                    csv = convert_df(dfsim)
                    st.download_button(
                        label="Export Label Studio csv",
                        data=csv,
                        file_name="label_studio_tasks.csv",
                        mime="text/csv",
                        key="ood_download"
                    )
            
            elif st.session_state.df_object.has_df_merge:
                st.warning("In order to run image visual similarity based outlier detection, you must upload a feature "
                        "vector file on 'Similarity search' tab")
            