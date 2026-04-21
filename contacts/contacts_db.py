import csv
import json
import matplotlib.pyplot as plt


class ContactNotFoundException(Exception):
    pass

def add_contact(contacts_db, first_name, last_name, phone, email):
    contact = {'first_name': first_name, 'last_name': last_name, 'phone': phone, 'email': email}
    contacts_db.append(contact)
    print(f"Contact added: {first_name} {last_name}")

def display_contacts(contacts):
    if not contacts:
        print("Address book is empty.")
    else:
        print("Contacts:")
        for contact in contacts:
            print(
                f"Name: {contact.get('first_name', '')} {contact.get('last_name', '')}, "
                f"Phone: {contact.get('phone', '')}, Email: {contact.get('email', '')}"
            )

def search_contact(contacts_db, first_name, last_name):
    for contact in contacts_db:
        if contact['first_name'] == first_name and contact['last_name'] == last_name:
            return contacts_db
    raise ContactNotFoundException(f"Contact not found for {first_name} {last_name}")

def plot_surname_distribution(contacts, save_plot=False):
    surnames = [contact.get('last_name', '') for contact in contacts if isinstance(contact, dict)]
    unique_surnames = list(set(surnames))
    surname_counts = [surnames.count(surname) for surname in unique_surnames]

    plt.bar(unique_surnames, surname_counts)
    plt.xlabel('Surnames')
    plt.ylabel('Number of Contacts')
    plt.title('Distribution of Contacts by Surname')

    if save_plot:
        plt.savefig('surname_distribution_plot.png')
        print("Plot saved as 'surname_distribution_plot.png'")
    else:
        plt.show()

def load_from_json_file(filename):
    try:
        with open(filename, 'r') as file:
            contacts_db = json.load(file)
        print(f"Contacts loaded from {filename} (JSON format)")
        return contacts_db
    except FileNotFoundError:
        print(f"File {filename} not found. Returning an empty list.")
        return []


def save_to_csv_file(contacts_db, filename):
    fieldnames = ['first_name', 'last_name', 'phone', 'email']
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(contacts_db)
    print(f"Contacts saved to {filename} (CSV format)")


def load_from_csv_file(filename):
    try:
        with open(filename, 'r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            contacts_db = list(reader)
        print(f"Contacts loaded from {filename} (CSV format)")
        return contacts_db
    except FileNotFoundError:
        print(f"File {filename} not found. Returning an empty list.")
        return []

def save_to_json_file(contacts_db, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(contacts_db, file)
    print(f"Contacts saved to {filename} (JSON format)")


if __name__ == "__main__":
    json_contacts_file = "contacts.json"
    csv_contacts_file = "contacts.csv"

    # Load contacts from JSON file (if the file exists)
    contacts = load_from_json_file(json_contacts_file)

    try:
        search_first_name = "Ανώνυμος"
        search_last_name = "Πολίτης"
        found_contact = search_contact(contacts, search_first_name, search_last_name)
        print(f"Contact found: {found_contact}")
    except ContactNotFoundException as e:
        print(f"Exception: {e}")

    # Adding contacts with Greek names and phone numbers
    add_contact(contacts, "Αλίκη", "Παπαδοπούλου", "+30 210 1234567", "alice@example.com")
    add_contact(contacts, "Βασίλης", "Κωστόπουλος", "+30 210 7654321", "vasilis@example.com")
    add_contact(contacts, "Ελένη", "Γεωργίου", "+30 210 9876543", "eleni@example.com")

    # Displaying contacts
    display_contacts(contacts)

    # Plotting surname distribution and saving the plot as an image
    plot_surname_distribution(contacts, save_plot=True)

    # Save contacts to JSON file
    save_to_json_file(contacts, json_contacts_file)

    # Save contacts to CSV file
    save_to_csv_file(contacts, csv_contacts_file)

    # Load contacts from CSV file
    loaded_contacts = load_from_csv_file(csv_contacts_file)
    display_contacts(loaded_contacts)
