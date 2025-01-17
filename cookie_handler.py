from http.server import BaseHTTPRequestHandler, HTTPServer

class MyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Check if a 'user_id' cookie exists
        cookies = self.headers.get('Cookie')
        user_id = None

        if cookies:
            # Parse cookies into a dictionary
            cookie_dict = dict(item.split("=") for item in cookies.split("; "))
            user_id = cookie_dict.get("user_id")
        
        if user_id:
            # User is recognized by their user_id
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(f"Welcome back, User ID: {user_id}".encode())
        else:
            # New user: Assign and store a user_id in a cookie
            user_id = "12345"  # Example: Static ID for testing (replace with dynamic ID generation if needed)
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.send_header("Set-Cookie", f"user_id={user_id}; Path=/; HttpOnly")
            self.end_headers()
            self.wfile.write(f"Hello, new user! Your User ID has been set to: {user_id}".encode())

# Run the server
port = 8080
server = HTTPServer(('localhost', port), MyHandler)
print(f"Server running on http://localhost:{port}")
server.serve_forever()
