# test_fase4.py — prueba el endpoint /api/process completo
# Ejecutar con Flask corriendo: python tests/test_fase4.py ruta\foto.jpg
import sys
import json
import urllib.request

def test_api(image_path: str):
    url     = "http://localhost:5000/api/process"
    boundary = "----FormBoundary"

    with open(image_path, "rb") as f:
        file_data = f.read()

    filename = image_path.split("\\")[-1]
    ext      = filename.split(".")[-1].lower()
    mime     = "image/jpeg" if ext in ("jpg","jpeg") else f"image/{ext}"

    body  = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {mime}\r\n\r\n"
    ).encode() + file_data + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST"
    )

    print(f"\nEnviando {filename} a {url} ...")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        data = json.loads(e.read())
        print(f"HTTP {e.code}: {data}")
        return

    print(f"Status:  {data.get('status')}")
    print(f"Detección: encontrado={data['detection']['found']}  "
          f"confianza={data['detection']['confidence']:.0%}")
    print(f"Landmarks: {data['landmarks']['count']} puntos")
    print(f"Normal map: {'OK' if data['normal_map']['map_b64'] else 'ERROR'}")
    print(f"Mesh: vértices={data['reconstruction']['mesh']['vertex_count']}  "
          f"caras={data['reconstruction']['mesh']['face_count']}")

    if data["pose"]["success"]:
        ang = data["pose"]["euler_angles"]
        print(f"Pose:    Yaw={ang['yaw']}°  "
              f"Pitch={ang['pitch']}°  "
              f"Roll={ang['roll']}°")
    print("\nFASE 4 — API REST completada correctamente.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python tests/test_fase4.py ruta\\a\\foto.jpg")
        sys.exit(1)
    test_api(sys.argv[1])