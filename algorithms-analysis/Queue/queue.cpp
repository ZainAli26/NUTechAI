#include <iostream>
using namespace std;

struct Node {
    int data;
    Node* next;
};

class Queue {
    Node* front;
    Node* rear;

public:
    Queue() {
        front = rear = NULL;
    }

    // Enqueue an element at the rear of the queue
    void enqueue(int data) {
        Node* newNode = new Node();
        newNode->data = data;
        newNode->next = NULL;

        if (rear == NULL) {
            front = rear = newNode;
            cout << "Enqueued: " << data << endl;
            return;
        }

        rear->next = newNode;
        rear = newNode;
        cout << "Enqueued: " << data << endl;
    }

    // Dequeue an element from the front of the queue
    void dequeue() {
        if (front == NULL) {
            cout << "Queue underflow!" << endl;
            return;
        }

        Node* temp = front;
        front = front->next;

        if (front == NULL) {
            rear = NULL;
        }

        cout << "Dequeued: " << temp->data << endl;
        delete temp;
    }

    // Peek the front element of the queue
    int peek() {
        if (front == NULL) {
            cout << "Queue is empty!" << endl;
            return -1;
        }
        return front->data;
    }

    // Check if the queue is empty
    bool isEmpty() {
        return front == NULL;
    }

    // Display the queue
    void display() {
        Node* temp = front;
        cout << "Queue: ";
        while (temp != NULL) {
            cout << temp->data << " ";
            temp = temp->next;
        }
        cout << endl;
    }
};

int main() {
    Queue queue;
    queue.enqueue(10);
    queue.enqueue(20);
    queue.enqueue(30);
    queue.display();
    queue.dequeue();
    queue.display();
    cout << "Front element: " << queue.peek() << endl;
    return 0;
}
