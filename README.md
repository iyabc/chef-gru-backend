# chef-blstm-backend

- run `python3 -m ensurepip --upgrade`
- run `python3 google-api-scraping/download.py -l https://drive.google.com/file/d/link-to-client-secret`
- run `python3 google-api-scraping/download.py -l https://drive.google.com/drive/folders/link-to-model-folder`
- run `pip3 install -r requirements.txt`
- . bin/activate
- run `python server.py`

to deploy backend:

- make sure ngrok is installed or ngrok file is in system32 (para pwde ra sya ma call in any terminal)
- open any terminal (pwde ra sa vscode new terminal ra)
- run ngrok http --domain=innocent-parrot-divine.ngrok-free.app http://127.0.0.1:8080 or run ngrok http --domain=innocent-parrot-divine.ngrok-free.app 80 (depende asa na port imong
 backend ga run)
- run backend locally