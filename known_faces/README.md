# Known Faces

This folder is how people are "enrolled" for recognition. There is no
separate enrollment script in this project - enrollment means adding a
photo here.

## Structure

```
known_faces/
    <person_name>/
        <any_image>.jpg
```

- One sub-folder per person. The folder name is used as the displayed
  label during recognition.
- Each sub-folder can contain one or more images of that person
  (`.jpg`, `.jpeg`, `.png`, `.jfif`, `.bmp`).
- Use clear, front-facing, well-lit photos. Each image should contain
  exactly one face - only the first detected face per photo is used.

## Example

```
known_faces/
    example_person/
        your_image.jpg
```

Replace `example_person` with a real name and `your_image.jpg` with an
actual photo before running recognition. The `example_person/` folder
in this repo is a placeholder only and contains no real image.
