# Data Download Instructions

The data files for this project are not included in the repository due to their size. Download the entire contents of the remote folder and place them in this `data/` directory.

## Download via SFTP

From the project root directory:

```bash
cd data/
sftp celftp@elbastion.dbs.umt.edu
```

Then download all files:

```bash
cd /celFtpFiles/pm_data
get *
exit
```

Or use a single command from the `data/` directory:

```bash
sftp celftp@elbastion.dbs.umt.edu:/celFtpFiles/pm_data/* .
```
