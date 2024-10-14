package com.example;

import com.vaadin.flow.component.button.Button;
import com.vaadin.flow.component.notification.Notification;
import com.vaadin.flow.router.Route;
import com.vaadin.flow.server.PWA;
import com.vaadin.flow.component.orderedlayout.VerticalLayout;

@Route("")
@PWA(
    name = "Vaadin PWA Example",
    shortName = "VaadinPWA",
    offlinePath = "offline.html",
    offlineResources = { "./images/offline.png" }
)
public class MainView extends VerticalLayout {

    public MainView() {
        Button button = new Button("Click me", event -> {
            Notification.show("Hello, Progressive Web App with Java!");
        });
        add(button);
    }
}
