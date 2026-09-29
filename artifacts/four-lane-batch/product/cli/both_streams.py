import sys,threading
def send(stream,letter):
    stream.write(letter*262144); stream.flush()
a=threading.Thread(target=send,args=(sys.stdout,'O'))
b=threading.Thread(target=send,args=(sys.stderr,'E'))
a.start(); b.start(); a.join(); b.join()
