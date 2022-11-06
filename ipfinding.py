import socket
import ipaddress

def ipdron():
    host_name = socket.gethostname()
    ipdres = socket.gethostbyname(host_name)

    print(host_name, ipdres)

    huh = 255

    bin_d = bin(huh)
    print(bin_d)

    print(f'{192:08b}')

    print('')

    ipa = int(ipaddress.IPv4Address(ipdres))

    bin_d = bin(ipa)
    print(bin_d)
    print(f'{ipa:08b}')

    