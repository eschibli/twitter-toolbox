from tqdm import tqdm
import requests

# Credit: https://tobiasraabe.github.io/blog/how-to-download-files-with-python.html#Visualizing-download-progress


def download_file(url, new_name):
    r = requests.get(url, stream=True)
    file_size = int(r.headers.get('content-length', 0))
    initial_pos = 0
    with open(new_name, 'wb') as f:
        with tqdm(total=file_size, unit='B', unit_scale=True, unit_divisor=1024, desc=new_name, initial=initial_pos,
                  ascii=True, miniters=1) as progress_bar:
            for chunk in r.iter_content(32 * 1024):
                f.write(chunk)
                progress_bar.update(len(chunk))

    
    



