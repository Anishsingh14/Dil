with open('app/main.py', 'r') as f:
    content = f.read()

content = content.replace('depth: str = "liveness"', 'depth: str = "readiness"')

with open('app/main.py', 'w') as f:
    f.write(content)

print('Done')