#include <iostream>
using namespace std;

struct Node {
    int data;
    Node* next;
};

class CircularLinkedList {
    Node* last;

public:
    CircularLinkedList() {
        last = NULL;
    }

    // Add a node at the end of the circular linked list
    void append(int data) {
        Node* newNode = new Node();
        newNode->data = data;

        // If the list is empty
        if (last == NULL) {
            last = newNode;
            last->next = last;  // Pointing to itself
        } else {
            newNode->next = last->next;  // newNode points to the first node
            last->next = newNode;        // Last node points to newNode
            last = newNode;              // Move the last pointer to the new node
        }
        cout << "Appended: " << data << endl;
    }

    // Add a node at the beginning of the circular linked list
    void prepend(int data) {
        Node* newNode = new Node();
        newNode->data = data;

        if (last == NULL) {
            last = newNode;
            last->next = last;  // Pointing to itself
        } else {
            newNode->next = last->next;  // newNode points to the first node
            last->next = newNode;        // Last node points to newNode
        }
        cout << "Prepended: " << data << endl;
    }

    // Remove a node by value
    void remove(int data) {
        if (last == NULL) {
            cout << "List is empty, cannot remove." << endl;
            return;
        }

        Node* current = last->next;
        Node* previous = last;

        // Traverse the list to find the node to delete
        do {
            if (current->data == data) {
                if (current == last && current->next == last) {
                    // Only one node in the list
                    last = NULL;
                } else if (current == last) {
                    // If the node to delete is the last node
                    previous->next = last->next;
                    last = previous;
                } else {
                    // Node is in the middle or at the beginning
                    previous->next = current->next;
                }
                delete current;
                cout << "Removed: " << data << endl;
                return;
            }
            previous = current;
            current = current->next;
        } while (current != last->next);

        cout << "Node with value " << data << " not found." << endl;
    }

    // Display the list
    void display() {
        if (last == NULL) {
            cout << "List is empty!" << endl;
            return;
        }

        Node* temp = last->next;
        cout << "List: ";
        do {
            cout << temp->data << " ";
            temp = temp->next;
        } while (temp != last->next);
        cout << endl;
    }
};

int main() {
    CircularLinkedList cll;

    cll.append(10);
    cll.append(20);
    cll.append(30);
    cll.display();

    cll.prepend(5);
    cll.display();

    cll.remove(20);
    cll.display();

    cll.remove(5);
    cll.display();

    cll.remove(30);
    cll.display();

    cll.remove(10);
    cll.display();

    return 0;
}
