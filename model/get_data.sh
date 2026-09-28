#!/usr/bin/env bash
# Downloads the goodbooks-10k ratings (CC BY-SA 4.0) next to this script.
set -e
cd "$(dirname "$0")"
curl -L -o ratings.csv https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/ratings.csv
curl -L -o books.csv https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/books.csv
