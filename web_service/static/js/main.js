$(document).ready(function () {
    $('#btn-start').attr('disabled', true);

    // Display the chosen file.
    $(".custom-file-input").on("change", function () {
        var fileName = $(this).val().split("\\").pop();
        if (fileName.length > 0) {
            $('#btn-start').removeAttr('disabled');
        } else {
            $('#btn-start').attr('disabled');
        }
        $(this).siblings(".custom-file-label").addClass("selected").html(fileName);
    });

    $('#audio-upload').change(function (e) {
        var fileName = e.target.files[0].name;
        if (fileName.length > 0) {
            $('#btn-start').removeAttr('disabled');
        } else {
            $('#btn-start').attr('disabled');
        }
    });

    // $('#btn-start').click(function () {
    //     // Make POST call to the server.
    //     $.post('/dance_figure', $('#upload-file').serialize()).done(function (data) {
    //         console.log(data);
    //     });
    // });
});