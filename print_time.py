from datetime import datetime
while True:
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S.%f")
    print(dt_string)