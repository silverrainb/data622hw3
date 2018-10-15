FROM python:3.7

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN git clone https://github.com/silverrainb/data622hw3 /usr/src/app/data622hw3

CMD [ "/usr/src/app/data622hw3/start.sh" ]