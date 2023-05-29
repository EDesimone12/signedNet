# Usa un'immagine di base con Python 2.7
FROM python:2.7

# Copia la cartella del progetto nell'immagine
COPY . /SignedNet

# Installa le dipendenze del progetto (ad es. SNAP)
RUN pip install snap-stanford

CMD ["/bin/bash"]


