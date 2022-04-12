"""
Created on Tue Nov  5 11:29:01 2019

@author: Tania Kleynhans

Connect to server, Run MODTRAN, and get tape6 back
Variables:  Hostname = 'tornado.cis.rit.edu'
            port = 22
            username = tkpci
            password = 
            filename = 'D:/Users/tkpci/Documents/DIRS_Research/MODTRAN/tape5/tape5'
"""

import paramiko
import os
from scp import SCPClient

def import_tape5_run_modtran(hostname='aldrin.cis.rit.edu',port=22,username='tkpci',password='Pr!ns3si1',
                             filename = '/cis/staff/tkpci/Code/data/tape5/tape5'):
    
    # below line for windows only
    # build the full path and convert it to RAW format using encode
    path_variable = os.path.normpath(os.environ['USERPROFILE'] + '\Documents\MobaXterm\home\.ssh\known_hosts')
    
    s = paramiko.SSHClient()
    s.load_system_host_keys(path_variable)
    s.connect(hostname, port, username, password)

    #command = 'mkdir modtran'
    #(stdin, stdout, stderr) = s.exec_command(command)

    scp = SCPClient(s.get_transport())
    scp.put(filename,remote_path='./modtran/tape5')
    scp.close()

    command = 'cd modtran\n' \
              'ln -s /dirs/pkg/Mod4v3r1/DATA\n' \
              '/dirs/pkg/Mod4v3r1/Mod4v3r1.exe'
    (stdin, stdout, stderr) = s.exec_command(command)

    s.close()

    print('Uploaded tape5 to server and running MODTRAN')

def export_tape6(hostname='aldrin.cis.rit.edu',port=22,username='tkpci',password='Pr!ns3si1',
                 tape6_filename='/cis/staff/tkpci/Code/data/tape6/tape6'):
    
    path_variable = os.path.normpath(os.environ['USERPROFILE'] + '\Documents\MobaXterm\home\.ssh\known_hosts')
    
    s = paramiko.SSHClient()
    s.load_system_host_keys(path_variable)
    s.connect(hostname, port, username, password)

    scp = SCPClient(s.get_transport())
    scp.get(remote_path='./modtran/tape6', local_path=tape6_filename)
    scp.close()

    #command = 'rm -r ./modtran/'
    #(stdin, stdout, stderr) = s.exec_command(command)
    s.close()

    print('Saved tape6 to file')