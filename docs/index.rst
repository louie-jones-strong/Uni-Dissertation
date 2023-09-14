.. DECAF documentation master file, created by
   sphinx-quickstart on Mon Jul  3 02:24:34 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to DECAF's documentation!
========================================



Quick Start
==================

1. Clone the repository


2. Run the following commands in the root directory

```
Setup.bat
```

```
docker-compose build
```

```
docker-compose up
```


3. Navigate to ```localhost:5000``` in your browser

You can modify the settings in the `.env` file.



Logging to wandb
------------
To log runs to wandb, create a file called `secrets.env` in the root directory with the following contents:

```
WANDB_API_KEY=<your api key>
```








.. toctree::
   :maxdepth: 4
   :caption: Contents:

   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
