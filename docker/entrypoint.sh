#!/bin/sh
# Apply database migrations once, before the server starts. The app's own
# auto-migration is disabled in the image so that several workers never race.
set -eu
groundedai migrate
exec "$@"
