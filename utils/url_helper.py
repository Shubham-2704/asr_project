"""
url_helper.py - URL Transformation Utility
==========================================
The original CSV has URLs pointing to:
    https://storage.googleapis.com/joshtalks-data-collection/hq_data/hi/{user_id}/{file}

But the assignment says the correct URLs follow this pattern:
    https://storage.googleapis.com/upload_goai/{user_id}/{file}

This module handles the URL transformation so all downloads work correctly.
"""

import re


# Old base URL in the CSV
OLD_BASE = "https://storage.googleapis.com/joshtalks-data-collection/hq_data/hi/"

# New base URL (from assignment instructions)
NEW_BASE = "https://storage.googleapis.com/upload_goai/"


def transform_url(url: str) -> str:
    """
    Transform a URL from the old CSV format to the new working format.

    Example:
        Old: https://storage.googleapis.com/joshtalks-data-collection/hq_data/hi/967179/825780_audio.wav
        New: https://storage.googleapis.com/upload_goai/967179/825780_audio.wav

    Args:
        url: Original URL from the CSV file

    Returns:
        Transformed URL pointing to the correct GCS bucket
    """
    if OLD_BASE in url:
        return url.replace(OLD_BASE, NEW_BASE)
    return url


def get_recording_id_from_url(url: str) -> str:
    """
    Extract the recording ID from a URL.

    Example:
        URL: .../967179/825780_audio.wav -> returns "825780"
    """
    filename = url.split("/")[-1]
    # Remove extension and suffix like _audio, _transcription, _metadata
    recording_id = re.sub(r"_(audio|transcription|metadata)\.\w+$", "", filename)
    return recording_id


def get_user_folder_from_url(url: str) -> str:
    """
    Extract the user folder ID from a URL.

    Example:
        URL: .../967179/825780_audio.wav -> returns "967179"
    """
    parts = url.rstrip("/").split("/")
    # The user folder is the second-to-last part
    return parts[-2]


if __name__ == "__main__":
    # Quick test
    old_url = "https://storage.googleapis.com/joshtalks-data-collection/hq_data/hi/967179/825780_audio.wav"
    new_url = transform_url(old_url)
    print(f"Old URL: {old_url}")
    print(f"New URL: {new_url}")
    print(f"Recording ID: {get_recording_id_from_url(old_url)}")
    print(f"User Folder: {get_user_folder_from_url(old_url)}")
