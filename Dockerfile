FROM polarize_base:latest

COPY . .
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
