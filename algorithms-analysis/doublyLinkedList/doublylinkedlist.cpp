#include <iostream>
using namespace std;

struct Node {
    int data;
    Node* next;
    Node* prev;
};

class DoublyLinkedList {
    Node* head;

public:
    // Constructor to initialize an empty list
    DoublyLinkedList() {
        head = NULL;
    }

    // Add a node at the end of the list
    void append(int data) {
        Node* newNode = new Node();
        newNode->data = data;
        newNode->next = NULL;
        newNode->prev = NULL;

        if (head == NULL) {
            head = newNode;  // First node
            return;
        }

        Node* temp = head;
        while (temp->next != NULL) {
            temp = temp->next;
        }

        temp->next = newNode;
        newNode->prev = temp;
        cout << "Appended: " << data << endl;
    }

    // Add a node at the beginning of the list
    void prepend(int data) {
        Node* newNode = new Node();
        newNode->data = data;
        newNode->next = head;
        newNode->prev = NULL;

        if (head != NULL) {
            head->prev = newNode;
        }

        head = newNode;
        cout << "Prepended: " << data << endl;
    }

    // Remove a node by value
    void remove(int data) {
        Node* temp = head;

        if (head == NULL) {
            cout << "List is empty, cannot remove." << endl;
            return;
        }

        // If the node to be deleted is the head node
        if (head->data == data) {
            head = head->next;
            if (head != NULL) {
                head->prev = NULL;
            }
            delete temp;
            cout << "Removed: " << data << endl;
            return;
        }

        // Traverse the list to find the node to delete
        while (temp != NULL && temp->data != data) {
            temp = temp->next;
        }

        if (temp == NULL) {
            cout << "Node with value " << data << " not found." << endl;
            return;
        }

        // If the node to be deleted is in the middle or end
        temp->prev->next = temp->next;
        if (temp->next != NULL) {
            temp->next->prev = temp->prev;
        }

        delete temp;
        cout << "Removed: " << data << endl;
    }

    // Display the list from the head (forward)
    void displayForward() {
        Node* temp = head;
        cout << "List (forward): ";
        while (temp != NULL) {
            cout << temp->data << " ";
            temp = temp->next;
        }
        cout << endl;
    }

    // Display the list from the tail (backward)
    void displayBackward() {
        if (head == NULL) {
            cout << "List is empty!" << endl;
            return;
        }

        Node* temp = head;
        while (temp->next != NULL) {
            temp = temp->next;
        }

        cout << "List (backward): ";
        while (temp != NULL) {
            cout << temp->data << " ";
            temp = temp->prev;
        }
        cout << endl;
    }
};

int main() {
    DoublyLinkedList dll;
    
    dll.append(10);
    dll.append(20);
    dll.append(30);
    dll.displayForward();

    dll.prepend(5);
    dll.displayForward();
    dll.displayBackward();

    dll.remove(20);
    dll.displayForward();
    dll.displayBackward();

    dll.remove(5);
    dll.displayForward();

    return 0;
}
