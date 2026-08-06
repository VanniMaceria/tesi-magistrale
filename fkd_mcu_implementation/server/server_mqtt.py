import paho.mqtt.client as mqtt

def on_message(client, userdata, msg):
    print(f"Messaggio ricevuto! Dimensione payload: {len(msg.payload)} byte")
    if len(msg.payload) == 101800:
        print("SUCCESSO: Il payload dei pesi è arrivato perfettamente integro!")
    client.disconnect()

client = mqtt.Client()
client.on_message = on_message
client.connect("10.0.0.100", 1883, 60)
client.subscribe("fl/client_0/weights")

print("In attesa dei pesi dall'ESP32...")
client.loop_forever()