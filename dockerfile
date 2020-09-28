FROM python:3.8-buster

COPY ./requirements_docker.txt .
RUN apt update && apt upgrade -y
RUN pip3 install -r requirements_docker.txt

# streamlit-specific commands
RUN mkdir -p /root/.streamlit
RUN bash -c 'echo -e "\
[general]\n\
email = \"\"\n\
" > /root/.streamlit/credentials.toml'
RUN bash -c 'echo -e "\
[server]\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
" > /root/.streamlit/config.toml'

COPY ./app/ /app/
WORKDIR /app/
EXPOSE 8501

ENTRYPOINT ["streamlit", "run"]
CMD ["app.py"]
