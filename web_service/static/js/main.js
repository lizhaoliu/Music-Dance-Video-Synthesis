$(document).ready(function () {
    $("#video").hide();
    $("#btn-start").attr("disabled", true);

    // Display the chosen file.
    $(".custom-file-input").on("change", function () {
        var fileName = $(this).val().split("\\").pop();
        if (fileName.length > 0) {
            $("#btn-start").removeAttr("disabled");
        } else {
            $("#btn-start").attr("disabled", true);
        }
        $(this).siblings(".custom-file-label").addClass("selected").html(fileName);
    });

    $("#audio-upload").change(function (e) {
        var fileName = e.target.files[0].name;
        if (fileName.length > 0) {
            $("#btn-start").removeAttr("disabled");
        } else {
            $("#btn-start").attr("disabled");
        }
    });

    // Make POST call to the server.
    $("#btn-start").click(function (e) {
        $("#video-banner").text("Processing...");
        $("#video").hide();

        $.ajax({
            xhr: function () {
                var xhr = new window.XMLHttpRequest();
                xhr.upload.addEventListener("progress", function (evt) {
                    if (evt.lengthComputable) {
                        var percentComplete = evt.loaded / evt.total;
                        $("#video-banner").text("Uploaded: " + percentComplete * 100 + "%");
                        console.log(percentComplete);
                        if (percentComplete === 1) {
                            $("#video-banner").text("Upload completed, server processing...")
                        }
                    }
                }, false);
                return xhr;
            },
            url: "/dance_figure",
            type: "POST",
            cache: false,
            data: new FormData($("#upload-file")[0]),
            processData: false,
            contentType: false,
        }).done(function (data) {
            $("#video-banner").text("Dance ready!");
            $("#video").show();
            var video = document.getElementById("video");
            video.src = "data:video/mp4;base64," + data;
            video.load();
            video.play();
        }).fail(function (rep) {
            console.log(rep);
        });
    });
});
