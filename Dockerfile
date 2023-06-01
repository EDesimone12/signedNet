# Usa un'immagine di base con Python 3.7
FROM python:2.7

# Copia la cartella del progetto nell'immagine
COPY . /SignedNet

# Installa le dipendenze del progetto (ad es. SNAP)
RUN pip install snap-stanford
RUN pip install networkx
RUN pip install matplotlib


CMD ["/bin/bash"]


