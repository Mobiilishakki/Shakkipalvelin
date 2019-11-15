FROM fedora:31

COPY . /App

WORKDIR App

RUN dnf update -y && useradd app && sudo -u app pip install --user -r requirements.txt && chown -R app:app .

USER app

EXPOSE 5000

CMD python app.py
