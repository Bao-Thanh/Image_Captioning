import { Component, OnInit } from '@angular/core';
import * as pbi from 'powerbi-client';

@Component({
  selector: 'app-caption-result',
  templateUrl: './caption-result.component.html',
  styleUrls: ['./caption-result.component.scss'],
})
export class CaptionResultComponent implements OnInit {

  constructor() { }

  ngOnInit() {
    this.showReport()
  }

  showReport() {
    // Report's Secured Token
    let accessToken = '<Your Access Token Key1>  ';

    // Embed URL
    let embedUrl = 'https://embedded.powerbi.com/appTokenReportEmbed?reportId=<Your embedReportID>';

    // Report ID
    let embedReportId = '<Your embedReportID>';

    // Define a Configuration used to describe the what and how to embed.
    // This object is used when calling powerbi.embed.
    // This also includes settings and options such as filters.
    
    let config = {
      type: 'report',
      accessToken: accessToken,
      embedUrl: embedUrl,
      id: embedReportId
    };

    // Grab the reference to the div HTML element that will host the report.
    let reportContainer = <HTMLElement>document.getElementById('reportContainer');

    // Embed the report and display it within the div container.
    let powerbi = new pbi.service.Service(pbi.factories.hpmFactory, pbi.factories.wpmpFactory, pbi.factories.routerFactory);
    let report = powerbi.embed(reportContainer, config);

    // Report.off removes a given event handler if it exists.
    report.off("loaded");

    // Report.on will add an event handler which prints to Log window.
    report.on("loaded", function () {
      console.log("Loaded");
    });
  }
}
