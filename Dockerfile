FROM jupyter/minimal-notebook:latest
ENV PYTHONPATH $PYTHONPATH:/Users/seth/development/DLFS_code/lincoln
RUN pip install numpy 
RUN pip install scikit-learn
#RUN pip install torch
WORKDIR /home/jovyan
EXPOSE 8888