import requests, os


def post_files(FILES: dict):
    files = dict()
    for file in FILES:
        if os.path.isdir(file):
            for filename in os.listdir(file):
                if os.path.isfile(os.path.join(file, filename)):
                    files[filename] = open(os.path.join(file, filename), 'rb')
        elif os.path.isfile(file):
            filename = file[file.rfind(os.sep) + 1:]
            files[filename] = open(file, 'rb')
    return files


def post(user: str, FILES: dict):
    data = dict()
    url = "https://www.zjunjie.top/process"
    session = requests.session()
    session.headers = {
        'User-Agent':
        'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/51.0.2704.63 Safari/537.36',
        'Referer':
        url
    }
    session.get(url + '?user=' + user)
    #session.get("https://www.zjunjie.top/gpu")
    token = session.cookies.get('csrftoken')
    data['csrfmiddlewaretoken'] = token
    data["user"] = user
    files = post_files(FILES)
    session.post(url=url,
                 data=data,
                 headers=session.headers,
                 cookies=session.cookies,
                 files=files)

    print('Result can be seen at "https://zjunjie.top/process?user={}"'.format(
        user))
