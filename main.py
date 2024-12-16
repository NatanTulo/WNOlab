import cv2
import numpy as np

def gaussian_blur(image, kernel_size=5, sigma=1.0):
    kernel = np.fromfunction(
        lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-(kernel_size-1)/2)**2 + (y-(kernel_size-1)/2)**2) / (2*sigma**2)),
        (kernel_size, kernel_size)
    )
    kernel /= np.sum(kernel)

    pad_size = kernel_size // 2
    padded_image = np.pad(image, [(pad_size, pad_size), (pad_size, pad_size), (0, 0)], mode='reflect')

    # Splot przy użyciu numpy bez pętli
    shape = (image.shape[0], image.shape[1], kernel_size, kernel_size, image.shape[2])
    strides = (padded_image.strides[0], padded_image.strides[1], padded_image.strides[0], padded_image.strides[1], padded_image.strides[2])
    patches = np.lib.stride_tricks.as_strided(padded_image, shape=shape, strides=strides)
    blurred_image = np.tensordot(patches, kernel, axes=([2, 3], [0, 1]))

    return blurred_image.astype(image.dtype)

def connected_components_stats(binary_image):
    # Implementacja algorytmu etykietowania składowych połączonych przy użyciu numpy
    labels = np.zeros_like(binary_image, dtype=int)
    label = 0  # Zaczynamy od etykiety 0 dla tła
    stats = []

    # Przeszukiwanie obrazu
    for y in range(binary_image.shape[0]):
        for x in range(binary_image.shape[1]):
            if binary_image[y, x] == 255 and labels[y, x] == 0:
                label += 1  # Nowa etykieta dla nowej składowej
                # Inicjalizacja parametrów bieżącej składowej
                min_x, max_x = x, x
                min_y, max_y = y, y
                area = 1
                stack = [(y, x)]
                labels[y, x] = label

                while stack:
                    cy, cx = stack.pop()
                    # Sprawdzanie sąsiadów
                    for ny in [cy - 1, cy, cy + 1]:
                        for nx in [cx - 1, cx, cx + 1]:
                            if (0 <= ny < binary_image.shape[0] and 0 <= nx < binary_image.shape[1]):
                                if binary_image[ny, nx] == 255 and labels[ny, nx] == 0:
                                    labels[ny, nx] = label
                                    stack.append((ny, nx))
                                    area += 1
                                    min_x = min(min_x, nx)
                                    max_x = max(max_x, nx)
                                    min_y = min(min_y, ny)
                                    max_y = max(max_y, ny)
                # Dodanie statystyk bieżącej składowej
                stats.append([min_x, min_y, max_x - min_x + 1, max_y - min_y + 1, area])

    # Dodanie statystyk dla tła (etykieta 0)
    background_stats = [0, 0, 0, 0, 0]
    stats.insert(0, background_stats)

    num_labels = label + 1  # Liczba etykiet łącznie z tłem
    stats = np.array(stats)
    return num_labels, labels, stats, None

def find_and_draw_differences(original_image_path, edited_image_path, output_path, largest_bbox_output, x):
    # Wczytanie obrazów przed rozmyciem
    original = cv2.imread(original_image_path)
    edited = cv2.imread(edited_image_path)

    if original is None or edited is None:
        raise ValueError("Nie można wczytać jednego z obrazów. Sprawdź ścieżki.")

    # Rozmycie obrazów przy pomocy splotu z użyciem numpy (do wykrywania różnic)
    original_blur = gaussian_blur(original)
    edited_blur = gaussian_blur(edited)

    # Obliczenie różnicy między rozmytymi obrazami w przestrzeni RGB
    difference = np.abs(original_blur.astype(int) - edited_blur.astype(int)).sum(axis=2)

    # Binarizacja różnic
    binary_diff = (difference > 40).astype(np.uint8) * 255

    num_labels, labels, stats, _ = connected_components_stats(binary_diff)

    # Rysowanie bounding boxów na niezmodyfikowanym obrazie
    largest_area = 0
    largest_bbox = None
    bounding_box_count = 0

    for i in range(1, num_labels):  # Pomijamy tło (etykieta 0)
        x_coord, y_coord, w, h, area = stats[i]
        if area < x:
            continue  # Ignorujemy bounding boxy mniejsze niż x
        bounding_box_count += 1
        if area > largest_area:
            largest_area = area
            largest_bbox = (x_coord, y_coord, w, h)
        cv2.rectangle(edited, (x_coord, y_coord), (x_coord + w, y_coord + h), (0, 0, 255), 2)  # Czerwony prostokąt

    print(f"Liczba wykrytych bounding boxów: {bounding_box_count}")

    # Wycięcie największego bounding boxa i zapisanie go z obrazu przed rozmyciem
    if largest_bbox is not None:
        x_coord, y_coord, w, h = largest_bbox
        largest_bbox_image = edited[y_coord:y_coord+h, x_coord:x_coord+w]

        # Wyodrębnienie odpowiadającego regionu z oryginalnego obrazu
        largest_bbox_image_original = original[y_coord:y_coord+h, x_coord:x_coord+w]

        # Obliczenie różnicy między rozmytymi wersjami edytowanego a oryginalnego obrazu w wyciętym obszarze
        difference_bbox = np.abs(
            edited_blur[y_coord:y_coord+h, x_coord:x_coord+w].astype(int) - 
            original_blur[y_coord:y_coord+h, x_coord:x_coord+w].astype(int)
        ).sum(axis=2)

        # Tworzenie maski binarnej na podstawie różnicy
        mask = (difference_bbox > 40).astype(np.uint8)
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

        # Usunięcie tła z largest_bbox_image
        largest_bbox_image_no_bg = largest_bbox_image * mask

        # Przekształcenie obrazu do formatu z kanałem alfa (RGBA)
        largest_bbox_image_rgba = np.dstack((largest_bbox_image_no_bg, mask[:, :, 0] * 255))

        # Ustawienie kanału alfa na 0 tam, gdzie maska wynosi 0
        largest_bbox_image_rgba[:, :, 3] = mask[:, :, 0] * 255

        # Zapisanie obrazu z przezroczystym tłem
        cv2.imwrite(largest_bbox_output, largest_bbox_image_rgba)
        print(f"Największy bounding box zapisano jako {largest_bbox_output}")

    # Zapisanie obrazu z zaznaczonymi bounding boxami
    cv2.imwrite(output_path, edited)    
    print(f"Obraz z zaznaczonymi bounding boxami zapisano jako {output_path}")

# Przykład użycia z minimalnym obszarem x (np. x = 1000)
find_and_draw_differences("org.jpg", "edited.jpg", "differences_output.png", "largest_bbox_output.png", x=100)
