import os
import shutil
import subprocess
import paho.mqtt.client as mqtt
from dotenv import load_dotenv

load_dotenv()

# MQTT configurations from .env
broker_address = os.getenv("MQTT_BROKER", "localhost")
broker_port = int(os.getenv("MQTT_PORT", "1883"))
topic = os.getenv("MQTT_TOPIC", "sign_language/sentence")
default_language = os.getenv("DEFAULT_LANGUAGE", "en")
received_log_path = os.getenv("RECEIVED_TEXT_PATH", "received_text.txt")

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"Connected to MQTT broker at {broker_address}:{broker_port}")
        client.subscribe(topic)
    else:
        print(f"Failed to connect, return code {rc}")

def on_message(client, userdata, msg):
    message = msg.payload.decode('utf-8').strip()
    print(f"Received message: {message}")

    if '.' in message:
        sentence = message.split('.')[0].strip()
        print(f"Final Sentence: {sentence}")

        try:
            with open(received_log_path, 'w', encoding='utf-8') as file:
                file.write(sentence)
        except IOError as e:
            print(f"Failed to write sentence log: {e}")

        # Check if espeak is installed on the host
        if shutil.which("espeak"):
            try:
                # Use subprocess.run with argument list to prevent command injection
                subprocess.run(
                    ["espeak", "-v", default_language, sentence],
                    check=False
                )
            except Exception as e:
                print(f"Error executing espeak: {e}")
        else:
            print(f"[TTS Alert - espeak not installed]: Would speak: '{sentence}' in {default_language}")

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

try:
    print(f"Connecting to MQTT broker at {broker_address}:{broker_port}...")
    client.connect(broker_address, broker_port, 60)
    client.loop_forever()
except Exception as e:
    print(f"Error starting MQTT listener: {e}")
