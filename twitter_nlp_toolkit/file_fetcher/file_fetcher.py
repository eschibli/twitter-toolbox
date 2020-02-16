import wget
import time

class downloader:
    def __init__(self,using_notebook=False):
        self.using_notebook = using_notebook
        self.last_timestamp = time.time()
        self.last_download_value = 0
        print("downloader initialized")
        
    def bar_custom_notebook(self,current, total, width=80):
        from IPython.display import clear_output, display
        diff = (current/total)*100 - (self.last_download_value/total)*100
        
        if self.last_download_value == 0:
            print("Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total),"           \r")
        
      
        
        if diff > 0.3:
            clear_output(wait=False)
            print("Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total),"           \r") 
            self.last_download_value = current            
        
        # elapsed_time = time.time() - self.last_timestamp
        # if(elapsed_time > 0.1):
            # self.last_timestamp = time.time()
            # print(elapsed_time)
            # clear_output(wait=False)
            # print("Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total),"           \r")

        
        
    
    def bar_custom(self,current, total, width=80):
        print("Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total),"           \r")
    
    def download_file(self,url,new_name):
        if self.using_notebook:
            wget.download(url,new_name,bar=self.bar_custom_notebook)
        else:
            wget.download(url,new_name,bar=self.bar_custom)

    
    


