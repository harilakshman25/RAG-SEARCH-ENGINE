import argparse
from lib.multimodal_search import verify_image_embedding, image_search_command
from lib.search_utils import DATA_DIR

def main():
   parser = argparse.ArgumentParser(description="MultiModal Search")
   subparsers = parser.add_subparsers(dest="command")

   verify_parser = subparsers.add_parser("verify", help="Verify the image embedding")
   verify_parser.add_argument("image_path", type=str, help="Image Path w.r.t to 'data' directory for the embedding")

   image_search_parser = subparsers.add_parser("image_search", help="Search using an image")
   image_search_parser.add_argument("image_path", type=str, help="Image Path w.r.t to 'data' directory for the search")

   args = parser.parse_args()

   match args.command:
       case "verify":
           image_path = DATA_DIR + '/' + args.image_path
           verify_image_embedding(image_path)
       case "image_search":
           image_path = DATA_DIR + '/' + args.image_path
           image_search_command(image_path)
       case _:
           parser.print_help()

if  __name__ == "__main__":
    main()