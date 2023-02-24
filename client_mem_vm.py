import socket


HOST = '192.168.122.49'
PORT = 8080



class ClientMemVM():

    def __init__(self):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    def connect(self):
        self.client.connect((HOST, PORT))
        message ="I'm client"
        n = self.client.send(message.encode())
        if (n != len(message)):
            print('Erreur envoi.')
        else: 
            print("Connexion succesfull")
    
    def get_server(self):
        message ="GET"
        n = self.client.send(message.encode())
        if (n != len(message)):
                print('Erreur envoi.')


    def get_value(self):
        self.get_server()
        donnees = self.client.recv(1024)
        return int(donnees.decode())

    def close_client(self):
        self.client.close()


