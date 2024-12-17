# Using BioDiscover Studio

Run the help command with
```bash
taxonomist-studio --help
```

Possible commands:

- `ppp`: Runs the Plate Position Paster
- `mdma`: Runs the Meta Data Managing Application
- `dataset`: Runs dataset commands

The `dataset` commands contain functions related to managing your BioDiscover image files. Possible commands are:

- `dataset scan --data_folder [image folder]`: Scans the images present on disk
- `dataset explore`: Starts a GUI for exploring the BioDiscover data

## Explore

Running

```bash
taxonomist-studio dataset explore
```

starts a GUI that makes it possible to explore the image data, with associated metadata.

### Key concepts:

**BioDiscover spreadsheet** is the file that is outputted from the SRS application

**Metadata spreadsheet** is an `.csv` or an `.xlsx` file that contains additional information (such as weights) on specimen.

**Image folder scan file** is the file output from `taxonomist-studio dataset scan`, and lists all files present on disk.