# db_client.py (Corrected for ClientOptions)

import os
from typing import TypedDict
from supabase import create_client, Client
from supabase.lib.client_options import ClientOptions  # <--- IMPORT THE FIX
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
SUPABASE_URL: str = os.environ.get("SUPABASE_URL")
SUPABASE_KEY: str = os.environ.get("SUPABASE_ANON_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Supabase URL and Key must be set in the .env file.")

# --- Type Hint for User Data ---
class AuthenticatedUser(TypedDict):
    user: object
    jwt: str

# --- Client Factory Function ---
def get_supabase_client(jwt: str = None) -> Client:
    """
    Creates a Supabase client.
    If a JWT is provided, it initializes the client with the user's auth context,
    allowing it to perform operations that require authentication (like INSERTs).
    Otherwise, it uses the public anon key for public operations (like login).
    """
    if jwt:
        # ### THE FIX IS HERE ###
        # We must use the ClientOptions class to pass headers correctly.
        options = ClientOptions(
            headers={"Authorization": f"Bearer {jwt}"}
        )
        return create_client(SUPABASE_URL, SUPABASE_KEY, options=options)
        # ### END OF FIX ###
    else:
        # Create a generic, unauthenticated client
        return create_client(SUPABASE_URL, SUPABASE_KEY)